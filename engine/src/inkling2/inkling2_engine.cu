// src/inkling2/inkling2_engine.cu - Standalone / integrated CUDA + AVX2 hybrid engine for Inkling-2-Mini (arch inkling2).
#include "strata/inkling2/engine.hpp"
#include "strata/inkling2/sampling.hpp"
#include "strata/inkling2/replica_policy.hpp"
#include "inkling2_kernels.cuh"
#include "inkling2_replica_async.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/types.hpp"
#if defined(STRATA_INKLING2_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::inkling2 {
namespace {

using Clock = std::chrono::steady_clock;
namespace K = strata::kernels;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

#define CK(call)                                                                                                  \
    do {                                                                                                          \
        cudaError_t err_ = (call);                                                                                \
        if (err_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "strata-inkling2 CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(err_));                                                               \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

#define CKB(call)                                                                                                 \
    do {                                                                                                          \
        cublasStatus_t err_ = (call);                                                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                                      \
            std::fprintf(stderr, "strata-inkling2 cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int) err_);  \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

// CUDA's current device is thread-local. Every runtime, weight allocation and destruction
// uses its owner explicitly; no peer access or UVA dereference is needed by the split path.
struct DeviceScope {
    int previous = -1;
    bool changed = false;
    explicit DeviceScope(int device) {
        if (device < 0) return;
        CK(cudaGetDevice(&previous));
        changed = previous != device;
        if (changed) CK(cudaSetDevice(device));
    }
    ~DeviceScope() { if (changed) cudaSetDevice(previous); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
};

// Opt-in replica storage: full residency is independent of compute ownership.
struct ExpertReplica {
    int device = -1;
    bool async = false;
    cudaEvent_t input_ready = nullptr, remote_done = nullptr;
    unsigned long long *frequencies = nullptr, *hits = nullptr; // primary-owned
    cudaStream_t stream = nullptr;
    std::vector<void*> allocs;
    std::vector<uint8_t*> blobs;
    std::vector<int32_t*> maps;
    std::vector<std::vector<int32_t>> cold_of;
    void *xq = nullptr, *scratch = nullptr;
    int32_t *ids = nullptr, *starts = nullptr, *groups = nullptr, *dst = nullptr, *tok = nullptr;
    unsigned long long* ptrs = nullptr;
    float* parts = nullptr;
    int32_t *host_ids = nullptr, *host_miss = nullptr;
    uint8_t* host_xq = nullptr;
    float* host_parts = nullptr;
    int32_t* local_miss = nullptr;
    float* local_parts = nullptr;
    int64_t local_hits = 0, remote_hits = 0, relay_bytes = 0;
    int64_t duplicate_entries = 0, missing_entries = 0;
    double join_ms = 0;
    size_t full_bytes = 0, hot_bytes = 0;
    template<class T> void alloc(T** p, size_t bytes) {
        CK(cudaMalloc((void**) p, bytes)); allocs.push_back(*p);
    }
    ~ExpertReplica() {
        DeviceScope owner(device);
        if (stream) cudaStreamSynchronize(stream);
        for (void* p : allocs) cudaFree(p);
        if (remote_done) cudaEventDestroy(remote_done);
        if (stream) cudaStreamDestroy(stream);
        cudaFreeHost(host_ids); cudaFreeHost(host_miss);
        cudaFreeHost(host_xq); cudaFreeHost(host_parts);
    }
};

struct Hparams {
    int64_t n_layer = 48;       // virtual layers
    int64_t n_phys = 24;        // physical layers
    int64_t num_loops = 3;
    int64_t core_start = 6;
    int64_t core_end = 18;
    int64_t n_embd = 2304;
    int64_t n_ff = 1152;
    int64_t n_head = 24;
    int64_t n_head_kv = 8;
    int64_t head_dim = 128;
    int64_t q_dim = 3072;
    int64_t kv_dim = 1024;
    int64_t n_expert = 128;
    int64_t n_expert_used = 8;
    int64_t n_rel = 16;
    int64_t n_rel_ext = 1024;
    int64_t n_swa = 1024;
    int64_t attn_res_block_size = 6;
    int64_t n_conv = 4;
    int64_t loop_gate_rank = 64;
    int64_t vocab_padded = 201024;
    int64_t vocab_unpadded = 200058;
    int64_t log_scale_floor = 8192;
    float log_scale_alpha = 0.1f;
    float rms_eps = 1e-6f;
    float qk_clip = 22.6274166f;
    float rel_clip = 8.0f;
    float situ_beta = 4.0f;
    float situ_linear_beta = 25.0f;
    float expert_weights_scale = 8.0f;
    float softcap = 30.0f;
    std::vector<uint8_t> swa_pattern;

    int64_t n_sconv_channels() const { return 2 * kv_dim + 2 * n_embd; }
    bool is_swa(int64_t il) const {
        if ((size_t) il < swa_pattern.size()) return swa_pattern[(size_t) il] != 0;
        return (il % 6) != 5;
    }
    int loop_layer(int il) const {
        if (num_loops <= 1) return il;
        const int64_t n_core = core_end - core_start;
        if (il < core_start) return il;
        if (il < core_start + num_loops * n_core) return (int) (core_start + ((il - core_start) % n_core));
        return (int) (il - (num_loops - 1) * n_core);
    }
};

struct DenseMat {
    int type = -1;
    int64_t n_in = 0;
    int64_t n_out = 0;
    void* d = nullptr;
    size_t bytes = 0;
};

struct PhysLayer {
    int device = -1;
    bool is_swa = true;
    bool has_loop = false;

    // Dense projections on GPU (BF16 or native GGUF quantized)
    DenseMat attn_q;          // [n_embd, q_dim]
    DenseMat attn_k;          // [n_embd, kv_dim]
    DenseMat attn_v;          // [n_embd, kv_dim]
    DenseMat attn_output;     // [q_dim, n_embd]
    DenseMat attn_gate;       // [n_embd, q_dim]
    DenseMat attn_sink_proj;  // [n_embd, n_head]
    DenseMat attn_rel;        // [n_embd, n_rel * n_head]
    DenseMat attn_qgr;        // concatenated [attn_q; attn_gate; attn_rel] for T==1 single MMVQ
    DenseMat attn_qgrkv;      // concatenated [attn_q; attn_gate; attn_rel; attn_k; attn_v] for SWA !has_msg T==1
    DenseMat attn_kkc;        // concatenated [attn_k; attn_k_conv_proj] for !is_swa T==1
    DenseMat attn_vvc;        // concatenated [attn_v; attn_v_conv_proj] for !is_swa T==1
    DenseMat attn_k_conv_proj; // [n_embd, n_conv * n_head_kv] (!is_swa)
    DenseMat attn_v_conv_proj; // [n_embd, n_conv * n_head_kv] (!is_swa)

    // Loop controllers [4]: 0=attn, 1=k, 2=v, 3=mlp
    DenseMat loop_down[4] = {};   // [2*n_embd, 64]
    DenseMat loop_gain[4] = {};   // [64, n_embd]
    DenseMat loop_forget[4] = {}; // [64, n_embd]

    // F32 params on GPU
    float* attn_norm = nullptr;          // SWA
    float* attn_q_inp_norm = nullptr;    // !is_swa
    float* attn_k_inp_norm = nullptr;    // !is_swa
    float* attn_v_inp_norm = nullptr;    // !is_swa
    float* attn_q_clip = nullptr;        // [head_dim, n_head]
    float* attn_rel_clip = nullptr;      // [n_rel, n_head]
    float* attn_rel_bias = nullptr;      // [n_rel_ext, n_rel] transposed -> wait: GGUF is [n_rel, n_ext], i.e. d*n_rel + r!
    float* attn_k_conv = nullptr;        // [4, kv_dim] -> c*4 + lag
    float* attn_v_conv = nullptr;
    float* attn_post_conv = nullptr;     // [4, n_embd]
    float* ffn_post_conv = nullptr;      // [4, n_embd]
    float* attn_res_query = nullptr;     // [n_embd, 1 or 3] (null on phys 0)
    float* ffn_res_query = nullptr;      // [n_embd]
    float* ffn_norm = nullptr;           // [n_embd]
    float* ffn_gate_inp = nullptr;       // [n_embd, n_expert]
    float* exp_probs_b = nullptr;        // [n_expert]

    // MoE experts (host mmap source + packed blobs + VRAM cache)
    int gu_type = -1;
    int d_type = -1;
    size_t gu_row = 0;
    size_t d_row = 0;
    const uint8_t* h_gate_up = nullptr;
    const uint8_t* h_down = nullptr;
    const float* h_gu_scale = nullptr;   // optional (NVFP4)
    const float* h_d_scale = nullptr;    // optional (NVFP4)

    K::NativeExpertLayout L{};
#if defined(STRATA_INKLING2_CPU_POOL)
    strata::kernels::cpu::NativeFmt cpu_fmt{};
#endif
    uint8_t* hblobs = nullptr;           // [n_expert][L.bytes] on host (lazy or pre-built)
    uint8_t* d_blobs = nullptr;          // [n_slots][stride] in VRAM
    size_t stride = 0;
    int n_slots = 0;
    int32_t* d_slot_of = nullptr;        // [n_expert] in VRAM
    std::vector<int32_t> slot_of;        // [n_expert] on host (-1 = CPU)
    std::vector<int32_t> exp_in_slot;    // [n_slots]
    std::vector<double> score;           // [n_expert]
};

struct GpuWeightStore {
    bool has_quant_dense = false;
    struct Allocation { void* p; int device; };
    std::vector<Allocation> gpu_allocs;
    std::vector<void*> host_allocs;
    std::vector<void*> host_pinned_allocs;
    std::vector<void*> q8_packed_keys;

    virtual ~GpuWeightStore() = default;
    void free_store() {
        for (void* p : q8_packed_keys) K::native_q8_0_packed_unregister(p);
        for (const auto& a : gpu_allocs) { DeviceScope owner(a.device); cudaFree(a.p); }
        for (void* p : host_pinned_allocs) cudaFreeHost(p);
        for (void* p : host_allocs) std::free(p);
        q8_packed_keys.clear(); gpu_allocs.clear(); host_pinned_allocs.clear(); host_allocs.clear();
    }

    template <typename T>
    T* gpu_alloc(size_t count) {
        T* p = nullptr;
        CK(cudaMalloc(&p, count * sizeof(T)));
        record_alloc(p);
        return p;
    }

    void record_alloc(void* p) {
        int device = 0;
        CK(cudaGetDevice(&device));
        gpu_allocs.push_back({p, device});
    }

    virtual const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const = 0;

    float* upload_f32(const char* name, size_t expected_elems = 0) {
        const strata::TensorInfo* t = nullptr;
        const uint8_t* raw = find_tensor(name, &t);
        if (!raw || !t) {
            std::fprintf(stderr, "strata-inkling2: missing tensor %s\n", name);
            std::exit(1);
        }
        const int64_t ne = (int64_t) t->elements();
        if (expected_elems && (size_t) ne != expected_elems) {
            std::fprintf(stderr, "strata-inkling2: tensor %s expected %zu elems got %lld\n", name, expected_elems, (long long) ne);
            std::exit(1);
        }
        float* d = gpu_alloc<float>((size_t) ne);
        if (t->type == 0) {
            CK(cudaMemcpy(d, raw, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice));
        } else if (t->type == 30) {
            std::vector<float> tmp((size_t) ne);
            const uint16_t* b = (const uint16_t*) raw;
            for (int64_t i = 0; i < ne; ++i) {
                uint32_t u = (uint32_t) b[i] << 16;
                std::memcpy(&tmp[(size_t) i], &u, 4);
            }
            CK(cudaMemcpy(d, tmp.data(), (size_t) ne * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            std::fprintf(stderr, "strata-inkling2: unsupported F32 param type %u on %s\n", t->type, name);
            std::exit(1);
        }
        return d;
    }

    DenseMat upload_dense(const char* name, size_t rows_override = 0, size_t cols_expected = 0,
                          bool pack_q8_in_place = false, bool pack_q8_extra = false) {
        const strata::TensorInfo* t = nullptr;
        const uint8_t* raw = find_tensor(name, &t);
        if (!raw || !t || (t->type != 0 && t->type != 1 && t->type != 30 && !K::native_mmvq_supported((int) t->type))) {
            std::fprintf(stderr, "strata-inkling2: missing/unsupported dense tensor %s (type=%u)\n", name, t ? t->type : 999u);
            std::exit(1);
        }
        const int64_t ne0 = (int64_t) t->shape[0];
        const int64_t ne1 = rows_override ? (int64_t) rows_override : (t->shape.size() > 1 ? (int64_t) t->shape[1] : 1);
        if (cols_expected && (size_t) ne0 != cols_expected) {
            std::fprintf(stderr, "strata-inkling2: %s ne0 expected %zu got %lld\n", name, cols_expected, (long long) ne0);
            std::exit(1);
        }
        const size_t rb = (t->type == 0) ? (size_t) ne0 * sizeof(float)
                        : (t->type == 1 || t->type == 30) ? (size_t) ne0 * sizeof(uint16_t)
                        : strata::quant::row_bytes((int) t->type, ne0);
        if (rb == 0) {
            std::fprintf(stderr, "strata-inkling2: %s invalid row bytes (type=%u ne0=%lld)\n", name, t->type, (long long) ne0);
            std::exit(1);
        }
        const size_t bytes = rb * (size_t) ne1;
        uint8_t* d = gpu_alloc<uint8_t>(bytes);
        if (t->type == 8 && pack_q8_in_place && K::native_q8_0_packed_eligible((int) ne0, (int) ne1)) {
            std::vector<uint8_t> packed(bytes);
            K::native_q8_0_pack_host(raw, packed.data(), (int) ne0, (int) ne1);
            CK(cudaMemcpy(d, packed.data(), bytes, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, d, (int) ne0, (int) ne1);
            q8_packed_keys.push_back(d);
        } else {
            CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
            if (t->type == 8 && pack_q8_extra && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible((int) ne0, (int) ne1)) {
                std::vector<uint8_t> packed(bytes);
                K::native_q8_0_pack_host(raw, packed.data(), (int) ne0, (int) ne1);
                uint8_t* pd = gpu_alloc<uint8_t>(bytes);
                CK(cudaMemcpy(pd, packed.data(), bytes, cudaMemcpyHostToDevice));
                K::native_q8_0_packed_register(d, pd, (int) ne0, (int) ne1);
                q8_packed_keys.push_back(d);
            }
        }
        if (t->type != 30 && t->type != 0 && t->type != 1) has_quant_dense = true;
        return DenseMat{(int) t->type, ne0, ne1, d, bytes};
    }
};


struct Model : GpuWeightStore {
    std::unique_ptr<strata::GgufModel> gguf;
    Hparams hp;
    int main_device = 0;
    int split_device = -1;     // < 0: the unchanged single-device placement

    int layer_device(int ip) const {
        return split_device >= 0 && hp.num_loops > 1 && ip >= hp.core_start && ip < hp.core_end
                   ? split_device : main_device;
    }

    const uint8_t* host_tok_embd = nullptr;    // [vocab_padded, n_embd] on host (mmap)
    int tok_embd_type = -1;
    size_t tok_embd_row_bytes = 0;
    float* tok_embd_norm = nullptr;
    float* loop_res_query = nullptr;
    float* loop_kv_res_query = nullptr;
    float* output_res_query = nullptr;
    float* output_norm = nullptr;
    DenseMat output_weight;                    // [vocab_unpadded, n_embd] in VRAM

    std::vector<PhysLayer> layers;
    ~Model() override { free_store(); }

    const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const override {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (out) *out = t;
        return t ? gguf->shard(shard).tensor_data(*t) : nullptr;
    }

    void dequant_tok(int32_t tok, float* dst) const {
        const int N = (int) hp.n_embd;
        const uint8_t* row = host_tok_embd + (size_t) tok * tok_embd_row_bytes;
        if (tok_embd_type == 30) {
            const uint16_t* bf = (const uint16_t*) row;
            for (int i = 0; i < N; ++i) {
                uint32_t u = (uint32_t) bf[i] << 16;
                std::memcpy(&dst[i], &u, 4);
            }
        } else if (tok_embd_type == 0) std::memcpy(dst, row, (size_t) N * sizeof(float));
        else strata::quant::dequant_row((strata::quant::Type) tok_embd_type, row, dst, N);
    }
};

#include "inkling2_dspark.cuh"

void build_expert_blob(const PhysLayer& L, int64_t e, uint8_t* dst) {
    const int64_t n_ff = L.L.n_ff, n_embd = L.L.n_embd;
    const size_t gu_expert_bytes = (size_t) (2 * n_ff) * L.gu_row;
    const size_t d_expert_bytes = (size_t) n_embd * L.d_row;
    const uint8_t* gu_src = L.h_gate_up + (size_t) e * gu_expert_bytes;
    const uint8_t* d_src = L.h_down + (size_t) e * d_expert_bytes;
    // [gate (n_ff rows) | up (n_ff rows) | down (n_embd rows)]
    std::memcpy(dst, gu_src, (size_t) n_ff * L.gu_row);
    std::memcpy(dst + L.L.up_off, gu_src + (size_t) n_ff * L.gu_row, (size_t) n_ff * L.gu_row);
    std::memcpy(dst + L.L.down_off, d_src, d_expert_bytes);
    if (L.L.tail_off) {
        const float sg = L.h_gu_scale ? L.h_gu_scale[e] : 1.0f;
        const float sd = L.h_d_scale ? L.h_d_scale[e] : 1.0f;
        const float tail[4] = {sg, sg, sd, 0.0f};
        std::memcpy(dst + L.L.tail_off, tail, sizeof tail);
    }
}

bool load_model(const std::string& path, Model& m, std::string& err) {
    DeviceScope owner(m.main_device);
    const auto t0 = Clock::now();
    try {
        m.gguf = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    const strata::GgufFile& meta = m.gguf->meta();
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    if (strata::model_family(meta) != strata::ModelFamily::Inkling2) {
        err = "expected architecture inkling2";
        return false;
    }
    auto get_i = [&](const char* k, int64_t def) -> int64_t {
        const auto* v = meta.get(k);
        return v && v->is_num() ? (int64_t) v->u : def;
    };
    auto get_f = [&](const char* k, float def) -> float {
        const auto* v = meta.get(k);
        return v ? (float) v->num() : def;
    };
    Hparams& hp = m.hp;
    hp.n_layer = get_i("inkling2.block_count", 48);
    hp.num_loops = get_i("inkling2.num_loops", 3);
    hp.core_start = get_i("inkling2.loop.core_start", 6);
    hp.core_end = get_i("inkling2.loop.core_end", 18);
    const int64_t n_core = hp.num_loops > 1 ? (hp.core_end - hp.core_start) : 0;
    hp.n_phys = hp.n_layer - (hp.num_loops - 1) * n_core;
    hp.n_embd = get_i("inkling2.embedding_length", 2304);
    hp.n_ff = get_i("inkling2.expert_feed_forward_length", get_i("inkling2.feed_forward_length", 1152));
    if (const auto* v = meta.get("inkling2.attention.head_count")) {
        hp.n_head = !v->items.empty() ? (int64_t) v->items[0].u : (int64_t) v->u;
    }
    if (const auto* v = meta.get("inkling2.attention.head_count_kv")) {
        hp.n_head_kv = !v->items.empty() ? (int64_t) v->items[0].u : (int64_t) v->u;
    }
    hp.head_dim = get_i("inkling2.attention.key_length", 128);
    hp.q_dim = hp.n_head * hp.head_dim;
    hp.kv_dim = hp.n_head_kv * hp.head_dim;
    hp.n_expert = get_i("inkling2.expert_count", 128);
    hp.n_expert_used = get_i("inkling2.expert_used_count", 8);
    hp.n_rel = get_i("inkling2.attention.relative_length", 16);
    hp.n_rel_ext = get_i("inkling2.attention.relative_extent", 1024);
    hp.n_swa = get_i("inkling2.attention.sliding_window", 1024);
    hp.attn_res_block_size = get_i("inkling2.attn_res.block_size", 6);
    hp.n_conv = get_i("inkling2.shortconv.l_cache", 4);
    hp.loop_gate_rank = get_i("inkling2.loop.gate_rank", 64);
    hp.vocab_padded = get_i("inkling2.vocab_size", 201024);
    hp.vocab_unpadded = get_i("inkling2.vocab_size_unpadded", 200058);
    hp.log_scale_floor = get_i("inkling2.attention.log_scale_floor", 8192);
    hp.log_scale_alpha = get_f("inkling2.attention.log_scale_alpha", 0.1f);
    hp.rms_eps = get_f("inkling2.attention.layer_norm_rms_epsilon", 1e-6f);
    hp.qk_clip = get_f("inkling2.attention.qk_clip", 22.6274166f);
    hp.rel_clip = get_f("inkling2.attention.relative_clip", 8.0f);
    hp.situ_beta = get_f("inkling2.activation.situ_beta", 4.0f);
    hp.situ_linear_beta = get_f("inkling2.activation.situ_linear_beta", 25.0f);
    hp.expert_weights_scale = get_f("inkling2.expert_weights_scale", 8.0f);
    hp.softcap = get_f("inkling2.final_logit_softcapping", 30.0f);
    if (const auto* v = meta.get("inkling2.attention.sliding_window_pattern")) {
        hp.swa_pattern.resize(v->items.size());
        for (size_t i = 0; i < v->items.size(); ++i) hp.swa_pattern[i] = v->items[i].u ? 1 : 0;
    }

    // Global tensors (token_embd kept in host mmap)
    const strata::TensorInfo* te_info = nullptr;
    m.host_tok_embd = m.find_tensor("token_embd.weight", &te_info);
    if (!m.host_tok_embd || !te_info) {
        err = "token_embd.weight missing";
        return false;
    }
    m.tok_embd_type = (int) te_info->type;
    if (m.tok_embd_type == 30) m.tok_embd_row_bytes = (size_t) hp.n_embd * sizeof(uint16_t);
    else if (m.tok_embd_type == 0) m.tok_embd_row_bytes = (size_t) hp.n_embd * sizeof(float);
    else if (strata::quant::can_dequant((strata::quant::Type) m.tok_embd_type)) {
        m.tok_embd_row_bytes = strata::quant::row_bytes(m.tok_embd_type, hp.n_embd);
    }
    if (m.tok_embd_row_bytes == 0) {
        err = "token_embd.weight unsupported type " + std::to_string(m.tok_embd_type);
        return false;
    }
    m.tok_embd_norm = m.upload_f32("token_embd_norm.weight", (size_t) hp.n_embd);
    if (hp.num_loops > 1) {
        DeviceScope core_owner(m.split_device >= 0 ? m.split_device : m.main_device);
        m.loop_res_query = m.upload_f32("loop_res_query.weight", (size_t) hp.n_embd * 2 * n_core);
        m.loop_kv_res_query = m.upload_f32("loop_kv_res_query.weight", (size_t) hp.n_embd * 2 * n_core);
    }
    m.output_res_query = m.upload_f32("output_res_query.weight", (size_t) hp.n_embd);
    m.output_norm = m.upload_f32("output_norm.weight", (size_t) hp.n_embd);
    const bool pack_head = std::getenv("STRATA_Q8_HEAD_PACK") != nullptr && std::getenv("STRATA_Q8_HEAD_PACK")[0] == '1';
    m.output_weight = m.upload_dense("output.weight", (size_t) hp.vocab_unpadded, (size_t) hp.n_embd, pack_head, false);

    m.layers.resize((size_t) hp.n_phys);
    char nm[128];
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        PhysLayer& L = m.layers[(size_t) ip];
        L.device = m.layer_device(ip);
        DeviceScope layer_owner(L.device);
        L.is_swa = hp.is_swa(ip);
        L.has_loop = (hp.num_loops > 1 && ip >= hp.core_start && ip < hp.core_end);
        const bool pack_core = L.has_loop;

        auto tn = [&](const char* suffix) -> const char* {
            std::snprintf(nm, sizeof nm, "blk.%d.%s", ip, suffix);
            return nm;
        };
        const char* nm_q = tn("attn_q.weight");
        const strata::TensorInfo* tq = nullptr;
        const uint8_t* raw_q = m.find_tensor(nm_q, &tq);
        const strata::TensorInfo *tg = nullptr, *tr = nullptr, *tk = nullptr, *tv = nullptr;
        std::string sn_g = "blk." + std::to_string(ip) + ".attn_gate.weight";
        std::string sn_r = "blk." + std::to_string(ip) + ".attn_rel.weight";
        std::string sn_k = "blk." + std::to_string(ip) + ".attn_k.weight";
        std::string sn_v = "blk." + std::to_string(ip) + ".attn_v.weight";
        const uint8_t* raw_g = m.find_tensor(sn_g.c_str(), &tg);
        const uint8_t* raw_r = m.find_tensor(sn_r.c_str(), &tr);
        const uint8_t* raw_k = m.find_tensor(sn_k.c_str(), &tk);
        const uint8_t* raw_v = m.find_tensor(sn_v.c_str(), &tv);
        const bool qgr_ok = raw_q && tq && raw_g && tg && raw_r && tr &&
                            tq->type == tg->type && tq->type == tr->type &&
                            tq->shape[0] == tg->shape[0] && tq->shape[0] == tr->shape[0] &&
                            tq->type != 0 && tq->type != 1 && tq->type != 30 && K::native_mmvq_supported((int) tq->type);
        const bool kv_ok_in_qgr = qgr_ok && L.is_swa && raw_k && tk && raw_v && tv &&
                                  tq->type == tk->type && tq->type == tv->type &&
                                  tq->shape[0] == tk->shape[0] && tq->shape[0] == tv->shape[0];
        if (kv_ok_in_qgr) {
            const int64_t nin = (int64_t) tq->shape[0];
            const size_t rb = strata::quant::row_bytes((int) tq->type, nin);
            const int64_t nq_out = tq->shape.size() > 1 ? (int64_t) tq->shape[1] : 1;
            const int64_t ng_out = tg->shape.size() > 1 ? (int64_t) tg->shape[1] : 1;
            const int64_t nr_out = tr->shape.size() > 1 ? (int64_t) tr->shape[1] : 1;
            const int64_t nk_out = tk->shape.size() > 1 ? (int64_t) tk->shape[1] : 1;
            const int64_t nv_out = tv->shape.size() > 1 ? (int64_t) tv->shape[1] : 1;
            const size_t bq = rb * (size_t) nq_out;
            const size_t bg = rb * (size_t) ng_out;
            const size_t br = rb * (size_t) nr_out;
            const size_t bk = rb * (size_t) nk_out;
            const size_t bv = rb * (size_t) nv_out;
            uint8_t* dall = m.gpu_alloc<uint8_t>(bq + bg + br + bk + bv);
            CK(cudaMemcpy(dall, raw_q, bq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq, raw_g, bg, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + bg, raw_r, br, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + bg + br, raw_k, bk, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dall + bq + bg + br + bk, raw_v, bv, cudaMemcpyHostToDevice));
            m.has_quant_dense = true;
            L.attn_q = DenseMat{(int) tq->type, nin, nq_out, dall, bq};
            L.attn_gate = DenseMat{(int) tg->type, nin, ng_out, dall + bq, bg};
            L.attn_rel = DenseMat{(int) tr->type, nin, nr_out, dall + bq + bg, br};
            L.attn_k = DenseMat{(int) tk->type, nin, nk_out, dall + bq + bg + br, bk};
            L.attn_v = DenseMat{(int) tv->type, nin, nv_out, dall + bq + bg + br + bk, bv};
            L.attn_qgr = DenseMat{(int) tq->type, nin, nq_out + ng_out + nr_out, dall, bq + bg + br};
            L.attn_qgrkv = DenseMat{(int) tq->type, nin, nq_out + ng_out + nr_out + nk_out + nv_out, dall, bq + bg + br + bk + bv};
        } else if (qgr_ok) {
            const int64_t nin = (int64_t) tq->shape[0];
            const size_t rb = strata::quant::row_bytes((int) tq->type, nin);
            const int64_t nq_out = tq->shape.size() > 1 ? (int64_t) tq->shape[1] : 1;
            const int64_t ng_out = tg->shape.size() > 1 ? (int64_t) tg->shape[1] : 1;
            const int64_t nr_out = tr->shape.size() > 1 ? (int64_t) tr->shape[1] : 1;
            const size_t bq = rb * (size_t) nq_out;
            const size_t bg = rb * (size_t) ng_out;
            const size_t br = rb * (size_t) nr_out;
            uint8_t* dqgr = m.gpu_alloc<uint8_t>(bq + bg + br);
            CK(cudaMemcpy(dqgr, raw_q, bq, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqgr + bq, raw_g, bg, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqgr + bq + bg, raw_r, br, cudaMemcpyHostToDevice));
            m.has_quant_dense = true;
            L.attn_q = DenseMat{(int) tq->type, nin, nq_out, dqgr, bq};
            L.attn_gate = DenseMat{(int) tg->type, nin, ng_out, dqgr + bq, bg};
            L.attn_rel = DenseMat{(int) tr->type, nin, nr_out, dqgr + bq + bg, br};
            L.attn_qgr = DenseMat{(int) tq->type, nin, nq_out + ng_out + nr_out, dqgr, bq + bg + br};
        } else {
            L.attn_q = m.upload_dense(tn("attn_q.weight"), 0, 0, false, pack_core);
            L.attn_gate = m.upload_dense(tn("attn_gate.weight"), 0, 0, false, pack_core);
            L.attn_rel = m.upload_dense(tn("attn_rel.weight"));
        }
        if (!kv_ok_in_qgr) {
            if (!L.is_swa) {
                const strata::TensorInfo *tkc = nullptr, *tvc = nullptr;
                std::string sn_kc = "blk." + std::to_string(ip) + ".attn_k_conv_proj.weight";
                std::string sn_vc = "blk." + std::to_string(ip) + ".attn_v_conv_proj.weight";
                const uint8_t* raw_kc = m.find_tensor(sn_kc.c_str(), &tkc);
                const uint8_t* raw_vc = m.find_tensor(sn_vc.c_str(), &tvc);
                auto upload_pair_k = [&](const uint8_t* ra, const strata::TensorInfo* ta, const uint8_t* rb_raw, const strata::TensorInfo* tb,
                                         DenseMat& out_a, DenseMat& out_b, DenseMat& out_pair, const char* nm_a, const char* nm_b) {
                    if (ra && ta && rb_raw && tb && ta->type == tb->type && ta->shape[0] == tb->shape[0] &&
                        ta->type != 0 && ta->type != 1 && ta->type != 30 && K::native_mmvq_supported((int) ta->type)) {
                        const int64_t nin = (int64_t) ta->shape[0];
                        const size_t rbytes = strata::quant::row_bytes((int) ta->type, nin);
                        const int64_t na = ta->shape.size() > 1 ? (int64_t) ta->shape[1] : 1;
                        const int64_t nb = tb->shape.size() > 1 ? (int64_t) tb->shape[1] : 1;
                        const size_t ba = rbytes * (size_t) na, bb = rbytes * (size_t) nb;
                        uint8_t* dpair = m.gpu_alloc<uint8_t>(ba + bb);
                        CK(cudaMemcpy(dpair, ra, ba, cudaMemcpyHostToDevice));
                        CK(cudaMemcpy(dpair + ba, rb_raw, bb, cudaMemcpyHostToDevice));
                        m.has_quant_dense = true;
                        out_a = DenseMat{(int) ta->type, nin, na, dpair, ba};
                        out_b = DenseMat{(int) tb->type, nin, nb, dpair + ba, bb};
                        out_pair = DenseMat{(int) ta->type, nin, na + nb, dpair, ba + bb};
                    } else {
                        out_a = m.upload_dense(nm_a, 0, 0, false, pack_core);
                        out_b = m.upload_dense(nm_b);
                    }
                };
                upload_pair_k(raw_k, tk, raw_kc, tkc, L.attn_k, L.attn_k_conv_proj, L.attn_kkc, tn("attn_k.weight"), sn_kc.c_str());
                upload_pair_k(raw_v, tv, raw_vc, tvc, L.attn_v, L.attn_v_conv_proj, L.attn_vvc, tn("attn_v.weight"), sn_vc.c_str());
            } else {
                L.attn_k = m.upload_dense(tn("attn_k.weight"), 0, 0, false, pack_core);
                L.attn_v = m.upload_dense(tn("attn_v.weight"), 0, 0, false, pack_core);
            }
        }
        L.attn_output = m.upload_dense(tn("attn_output.weight"), 0, 0, false, pack_core);
        L.attn_sink_proj = m.upload_dense(tn("attn_sink_proj.weight"));

        if (L.is_swa) {
            L.attn_norm = m.upload_f32(tn("attn_norm.weight"), (size_t) hp.n_embd);
        } else {
            L.attn_q_inp_norm = m.upload_f32(tn("attn_q_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_k_inp_norm = m.upload_f32(tn("attn_k_inp_norm.weight"), (size_t) hp.n_embd);
            L.attn_v_inp_norm = m.upload_f32(tn("attn_v_inp_norm.weight"), (size_t) hp.n_embd);
        }

        if (L.has_loop) {
            static const char* mods[4] = {"attn", "k", "v", "mlp"};
            for (int j = 0; j < 4; ++j) {
                char s0[64], s1[64], s2[64];
                std::snprintf(s0, sizeof s0, "loop_%s_down.weight", mods[j]);
                std::snprintf(s1, sizeof s1, "loop_%s_gain.weight", mods[j]);
                std::snprintf(s2, sizeof s2, "loop_%s_forget.weight", mods[j]);
                L.loop_down[j] = m.upload_dense(tn(s0));
                L.loop_gain[j] = m.upload_dense(tn(s1));
                L.loop_forget[j] = m.upload_dense(tn(s2));
            }
        }

        L.attn_q_clip = m.upload_f32(tn("attn_q_clip.weight"), (size_t) hp.head_dim * hp.n_head);
        L.attn_rel_clip = m.upload_f32(tn("attn_rel_clip.weight"), (size_t) hp.n_rel * hp.n_head);
        L.attn_rel_bias = m.upload_f32(tn("attn_rel_bias.weight"), (size_t) hp.n_rel * hp.n_rel_ext);
        L.attn_k_conv = m.upload_f32(tn("attn_k_conv.weight"), (size_t) hp.n_conv * hp.kv_dim);
        L.attn_v_conv = m.upload_f32(tn("attn_v_conv.weight"), (size_t) hp.n_conv * hp.kv_dim);
        L.attn_post_conv = m.upload_f32(tn("attn_post_conv.weight"), (size_t) hp.n_conv * hp.n_embd);
        L.ffn_post_conv = m.upload_f32(tn("ffn_post_conv.weight"), (size_t) hp.n_conv * hp.n_embd);
        if (ip > 0) L.attn_res_query = m.upload_f32(tn("attn_res_query.weight"), (size_t) hp.n_embd * (L.is_swa ? 1 : 3));
        L.ffn_res_query = m.upload_f32(tn("ffn_res_query.weight"), (size_t) hp.n_embd);
        L.ffn_norm = m.upload_f32(tn("ffn_norm.weight"), (size_t) hp.n_embd);
        L.ffn_gate_inp = m.upload_f32(tn("ffn_gate_inp.weight"), (size_t) hp.n_embd * hp.n_expert);
        L.exp_probs_b = m.upload_f32(tn("exp_probs_b.bias"), (size_t) hp.n_expert);

        // Experts
        const strata::TensorInfo *ti_gu = nullptr, *ti_d = nullptr, *ti_gus = nullptr, *ti_ds = nullptr;
        L.h_gate_up = m.find_tensor(tn("ffn_gate_up_exps.weight"), &ti_gu);
        L.h_down = m.find_tensor(tn("ffn_down_exps.weight"), &ti_d);
        if (!L.h_gate_up || !L.h_down || !ti_gu || !ti_d) {
            err = std::string("missing expert tensors in layer ") + std::to_string(ip);
            return false;
        }
        L.gu_type = (int) ti_gu->type;
        L.d_type = (int) ti_d->type;
        L.h_gu_scale = (const float*) m.find_tensor(tn("ffn_gate_up_exps.scale"), &ti_gus);
        L.h_d_scale = (const float*) m.find_tensor(tn("ffn_down_exps.scale"), &ti_ds);

        if (L.gu_type == 30 && L.d_type == 30) {
            L.L.gu_type = 30;
            L.L.d_type = 30;
            L.L.n_embd = hp.n_embd;
            L.L.n_ff = hp.n_ff;
            L.L.gu_row = (size_t) hp.n_embd * 2;
            L.L.d_row = (size_t) hp.n_ff * 2;
            L.L.up_off = (size_t) hp.n_ff * L.L.gu_row;
            L.L.down_off = 2 * L.L.up_off;
            L.L.bytes = L.L.down_off + (size_t) hp.n_embd * L.L.d_row;
            L.L.tail_off = 0;
        } else {
            if (!K::native_expert_supported(L.gu_type, L.d_type, hp.n_embd, hp.n_ff)) {
                err = "unsupported expert format in layer " + std::to_string(ip);
                return false;
            }
            L.L = K::native_expert_layout(L.gu_type, L.d_type, hp.n_embd, hp.n_ff);
        }
        L.L.situ = true;
        L.L.situ_beta = hp.situ_beta;
        L.L.situ_linear_beta = hp.situ_linear_beta;
        L.gu_row = L.L.gu_row;
        L.d_row = L.L.d_row;
        L.stride = (L.L.bytes + 255) & ~(size_t) 255;

#if defined(STRATA_INKLING2_CPU_POOL)
        if (!strata::kernels::cpu::native_fmt(L.gu_type, L.d_type, hp.n_embd, hp.n_ff, L.cpu_fmt, err)) {
            return false;
        }
        L.cpu_fmt.situ = true;
        L.cpu_fmt.situ_beta = hp.situ_beta;
        L.cpu_fmt.situ_linear_beta = hp.situ_linear_beta;
#endif
    }
    std::fprintf(stderr, "strata-inkling2: dense + metadata loaded in %.2f s (n_phys=%lld, n_virt=%lld)\n",
                 ms_since(t0) / 1000.0, (long long) hp.n_phys, (long long) hp.n_layer);
    return true;
}

struct Engine {
    Model* m = nullptr;
    enum class Role { All, Outer, Core };
    Role role = Role::All;
    int device = -1;
    Engine* split_peer = nullptr;       // non-owning; the two runtimes share one Model
    float* split_handoff = nullptr;     // portable pinned host memory, owned by Outer
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    std::unique_ptr<strata::prefill::Gemm> gemm;

    bool owns_phys(int ip) const {
        const bool core = m->hp.num_loops > 1 && ip >= m->hp.core_start && ip < m->hp.core_end;
        return role == Role::All || (role == Role::Core ? core : !core);
    }
    bool owns_virtual(int il) const { return owns_phys(m->hp.loop_layer(il)); }
    Engine& runtime_for(int il) { return !owns_virtual(il) && split_peer ? *split_peer : *this; }
    bool any_hybrid() const { return hybrid || (role == Role::Outer && split_peer && split_peer->hybrid); }

    int ctx = 16384;
    int tmax = 512;
    int64_t pos = 0;
    int32_t pos0_host = 0;      // survives begin_forward's asynchronous upload until finish_head synchronizes

    // KV caches & shortconv state (per virtual layer il = 0..47)
    bool kv_q8 = true;
    std::vector<__half*> kc, vc;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;
    std::vector<int> kv_cap;
    std::vector<float*> conv_hist;   // [48][n_sconv_channels * 3]

    // DSpark is allocated on the outer/head device; the core only captures its local taps/undo state.
    DsparkEngine* dspark = nullptr;
    int verify_max_t = 1;
    size_t layer_conv_floats = 0, local_conv_floats = 0;
    std::vector<int> trace_slot;
    SwaLayerPtrs swa_ptrs{};
    int swa_ring_cap = 0;
    uint8_t* swa_bak = nullptr;
    float* d_conv_trace = nullptr;
    float* d_tgt_feat[4] = {};
    float* split_features = nullptr;   // portable host staging for the two core taps

    // Activation scratch (all allocated up to tmax rows)
    float *cur = nullptr, *ffn_inp = nullptr;
    float *in_q = nullptr, *in_k = nullptr, *in_v = nullptr, *in_ffn = nullptr;
    float *msg0 = nullptr, *msg1 = nullptr, *msg2 = nullptr, *msg3 = nullptr;
    float *ctrl2N = nullptr, *ctrl64 = nullptr, *ctrl_g = nullptr, *ctrl_f = nullptr;
    float *qkv_ctrl2N = nullptr, *qkv_ctrl64 = nullptr; // opt-in T=1, three independent controllers
    float *norm_cur = nullptr;       // [tmax * N]
    float *prelude_cur_buf = nullptr;
    float *norm_blk_buf[9] = {};     // cached RMSNorm of pushed block residuals (0..7 + 8=bank_last)

    float *Q = nullptr, *K = nullptr, *V = nullptr, *R = nullptr;
    float *qgr_t1 = nullptr;         // [2 * q_dim + n_rel * n_head] contiguous T=1 fused output
    float *K_conv = nullptr, *V_conv = nullptr, *dyn_k = nullptr, *dyn_v = nullptr;
    float *sink = nullptr, *rel_tab = nullptr, *att = nullptr, *att_gate = nullptr, *wo = nullptr;
    float *attn_split = nullptr;     // [kAttnSplit * n_head * 130] for split GQA-3 decode
    float *router_logits = nullptr, *ewts = nullptr, *eparts = nullptr, *moe_raw = nullptr;
    float *logits = nullptr;
    float* h_tok_buf = nullptr;      // pinned [tmax * N] FP32 for token embedding upload
    uint16_t* act_bf16 = nullptr;    // [tmax * 2*N] (reused as BF16 or FP16 scratch)

    void *xq = nullptr, *xq_k = nullptr, *xq_v = nullptr, *escr = nullptr;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int32_t *tok_d = nullptr, *pos0_d = nullptr, *pick_d = nullptr;

    // Hybrid CPU pool staging
    int32_t* h_ids = nullptr;        // [tmax * top_k]
    float* h_h = nullptr; int cpu_rows = 0;            // [tmax * N]
    float* cpu_out = nullptr;        // [tmax * top_k * N]
    int32_t* h_miss = nullptr;       // [tmax * top_k]
    int32_t* d_miss = nullptr;
    float* ecpu = nullptr;           // [tmax * top_k * N]
    uint8_t* acts = nullptr;         // [tmax * kNativeActBytes]
    cudaEvent_t ev_route = nullptr;
    bool hybrid = false;

    // Double-buffered PCIe GPU streaming for prefill on hybrid layers
    struct PpRun {
        int16_t e0 = 0;
        int16_t cnt = 0;
        int16_t dst_off = 0;
    };
    struct PpTask {
        int ip = -1;
        int n_staged = 0;
        PpStageMap map{};
        std::vector<PpRun> runs;
    };
    cudaStream_t cst = nullptr;
    uint8_t* pp_stage[2] = {nullptr, nullptr};
    size_t pp_stage_bytes = 0;
    cudaEvent_t ev_pp_copy[2] = {nullptr, nullptr};
    cudaEvent_t ev_pp_used[2] = {nullptr, nullptr};
    cudaEvent_t ev_rebal = nullptr;
    bool has_pending_rebal = false;
    std::vector<PpTask> pp_tasks;
    std::vector<int> pp_layer_cnt;
    size_t pp_take_idx = 0;
    size_t pp_kick_idx = 0;
    int pp_stream_t = 16;
    bool pp_streaming = false;

#if defined(STRATA_INKLING2_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif

    std::unique_ptr<ExpertReplica> replica;

    // Stats
    int64_t hy_gpu = 0, hy_cpu = 0;
    double hy_cpu_ms = 0.0;
    int placed_slots = 0;
    size_t placed_bytes = 0;
    size_t vram_free_after_place = 0;
    int pool_workers = 0;

    void collect_replica_stats(bool scores) {
        auto& r = *replica;
        if (!r.async) return;
        DeviceScope owner(device);
        CK(cudaStreamSynchronize(st));
        unsigned long long hits[3];
        CK(cudaMemcpy(hits, r.hits, sizeof(hits), cudaMemcpyDeviceToHost));
        if (hits[2]) { std::fprintf(stderr, "strata-inkling2: async replica invalid router ID\n"); std::exit(1); }
        r.local_hits = (int64_t) hits[0]; r.remote_hits = (int64_t) hits[1];
        if (scores) {
            std::vector<unsigned long long> counts((size_t) m->hp.n_phys * m->hp.n_expert);
            CK(cudaMemcpy(counts.data(), r.frequencies, counts.size() * sizeof(counts[0]), cudaMemcpyDeviceToHost));
            for (size_t ip = 0; ip < m->layers.size(); ++ip)
                for (int e = 0; e < m->hp.n_expert; ++e) m->layers[ip].score[e] = (double) counts[ip * m->hp.n_expert + e];
        }
    }

    void print_replica_stats(const char* event) {
        if (!replica) return;
        collect_replica_stats(false);
        char join_value[64];
        std::snprintf(join_value, sizeof(join_value), replica->async ? "null" : "%.3f", replica->join_ms);
        std::fprintf(stderr, "REPLICA {\"event\":\"%s\",\"local_hits\":%lld,\"remote_hits\":%lld,\"computed_entries\":%lld,\"duplicate_entries\":%lld,\"missing_entries\":%lld,\"ownership_scope\":\"%s\",\"relay_bytes\":%lld,\"join_ms\":%s,\"join_timing_scope\":\"%s\",\"scheduling_mode\":\"%s\"}\n",
            event, (long long) replica->local_hits, (long long) replica->remote_hits,
                (long long) (replica->local_hits + replica->remote_hits),
                (long long) replica->duplicate_entries, (long long) replica->missing_entries,
                replica->async ? "validated_complement_maps_and_gpu_routes" : "per_entry_plan",
                (long long) replica->relay_bytes, join_value,
                replica->async ? "not_measured_async" : "host_schedule_and_join",
                replica->async ? "async_events" : "sync_host");
    }

    void reset_hybrid_stats() {
        hy_gpu = hy_cpu = 0; hy_cpu_ms = 0;
        if (replica) {
            replica->local_hits = replica->remote_hits = replica->relay_bytes = 0;
            replica->join_ms = 0;
            replica->duplicate_entries = replica->missing_entries = 0;
            if (replica->async) { DeviceScope owner(device); CK(cudaMemsetAsync(replica->hits, 0, 3 * sizeof(unsigned long long), st)); }
        }
        if (role == Role::Outer && split_peer) split_peer->reset_hybrid_stats();
    }
    int64_t total_hy_gpu() const { return hy_gpu + (role == Role::Outer && split_peer ? split_peer->hy_gpu : 0); }
    int64_t total_hy_cpu() const { return hy_cpu + (role == Role::Outer && split_peer ? split_peer->hy_cpu : 0); }
    double total_hy_cpu_ms() const { return hy_cpu_ms + (role == Role::Outer && split_peer ? split_peer->hy_cpu_ms : 0); }

    bool prof = false;
    bool prof_detail = false;
    double p_embd_ms = 0, p_attn_ms = 0, p_moe_gpu_ms = 0, p_moe_wait_ms = 0, p_moe_cpu_ms = 0;
    double p_moe_up_ms = 0, p_moe_post_ms = 0, p_core_ms = 0, p_head_ms = 0, p_rebal_ms = 0;
    double p_d_mix = 0, p_d_qkv = 0, p_d_attn = 0, p_d_out = 0, p_d_route = 0, p_d_moe_gpu_core = 0, p_d_moe_gpu_nv = 0, p_d_moe_gpu_q5 = 0;
    double p_d_cpu_nv = 0, p_d_cpu_q5 = 0;
    int64_t p_miss_nv = 0, p_miss_q5 = 0, p_hit_nv = 0, p_hit_q5 = 0;
    int p_calls = 0;
    void reset_prof() {
        if (role == Role::Outer && split_peer) split_peer->reset_prof();
        p_embd_ms = p_attn_ms = p_moe_gpu_ms = p_moe_wait_ms = p_moe_cpu_ms = 0;
        p_moe_up_ms = p_moe_post_ms = p_core_ms = p_head_ms = p_rebal_ms = 0;
        p_d_mix = p_d_qkv = p_d_attn = p_d_out = p_d_route = 0;
        p_d_moe_gpu_core = p_d_moe_gpu_nv = p_d_moe_gpu_q5 = 0;
        p_d_cpu_nv = p_d_cpu_q5 = 0;
        p_miss_nv = p_miss_q5 = p_hit_nv = p_hit_q5 = 0;
        p_calls = 0;
    }
    void print_prof(const char* tag) const {
        const double outer_moe = p_moe_gpu_ms + p_moe_wait_ms + p_moe_cpu_ms + p_moe_up_ms + p_moe_post_ms;
        const double pure_outer_attn = p_attn_ms;
        const double sum = p_embd_ms + p_core_ms + pure_outer_attn + outer_moe + p_head_ms + p_rebal_ms;
        std::fprintf(stderr,
                     "PROFILE %-4s (%d calls, sum=%.1f ms): embd=%.1f core=%.1f attn(outer)=%.1f "
                     "moe_gpu=%.1f moe_wait=%.1f moe_cpu=%.1f moe_up=%.1f moe_post=%.1f head=%.1f rebal=%.1f\n",
                     tag, p_calls, sum, p_embd_ms, p_core_ms, pure_outer_attn,
                     p_moe_gpu_ms, p_moe_wait_ms, p_moe_cpu_ms, p_moe_up_ms, p_moe_post_ms, p_head_ms, p_rebal_ms);
        if (prof_detail) {
            std::fprintf(stderr,
                         "  DETAIL %-4s: mix_ctrl=%.1f qkv_proj=%.1f kv_attn=%.1f out_conv=%.1f route=%.1f "
                         "moe_gpu(core=%.1f nv=%.1f q5=%.1f) moe_cpu(nv=%.1f/%lldmiss_%lldhit q5=%.1f/%lldmiss_%lldhit)\n",
                         tag, p_d_mix, p_d_qkv, p_d_attn, p_d_out, p_d_route,
                         p_d_moe_gpu_core, p_d_moe_gpu_nv, p_d_moe_gpu_q5,
                         p_d_cpu_nv, (long long) p_miss_nv, (long long) p_hit_nv,
                         p_d_cpu_q5, (long long) p_miss_q5, (long long) p_hit_q5);
        }
        if (role == Role::Outer && split_peer) split_peer->print_prof("core");
    }

    // Cached decode CUDA Graph for the contiguous 36-layer core (il = 6..41) when blk.6..17 are 100% in VRAM,
    // plus per-layer pre-route graphs (attn + mix + ffn_norm + route).
    cudaGraphExec_t core_graph = nullptr;
    cudaGraphExec_t capture_core_graph = nullptr;  // T=1 with DSpark feature taps
    cudaGraphExec_t verify_core_graph = nullptr;   // fixed verify_max_t, feature taps and conv trace
    cudaGraphExec_t outer_graph[2] = {};          // no-draft T=1 prefix / suffix, head excluded
    bool outer_graph_failed[2] = {};
    std::vector<cudaGraphExec_t> layer_pre_graph;
    bool core_all_gpu = false;
    bool split_core_graph = false;     // opt-in for the Core runtime; All keeps its existing graph policy
    bool batch_qkv_control_t1 = false; // fixed before init/capture; all virtual owners receive it
    bool split_outer_graph = false;
    bool outer_graph_profile_block = false;
    bool graph_oom = false;

    bool outer_graph_eligible() const {
        if (role != Role::Outer || !split_peer || dspark || outer_graph_profile_block || prof || prof_detail) return false;
        for (int ip = 0; ip < m->hp.n_phys; ++ip) if (owns_phys(ip)) {
            const auto& layer = m->layers[(size_t) ip];
            if (layer.device != device || layer.n_slots != m->hp.n_expert) return false;
        }
        return true;
    }

    // Prefix cache checkpoint (stores conv_hist + circular SWA KV state in host RAM; full-attn KV stays append-only on GPU)
    struct StateCheckpoint {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> data;
        std::vector<uint8_t> dspark_data;
    };

    StateCheckpoint save_checkpoint(const std::vector<int32_t>& tokens) {
        const auto& hp = m->hp;
        const size_t conv_bytes = (size_t) hp.n_sconv_channels() * 3 * sizeof(float);
        size_t total = (size_t) hp.n_layer * conv_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            if (kv_q8) total += cap * (size_t) hp.kv_dim * 2 + cap * (size_t) hp.n_head_kv * sizeof(__half) * 2;
            else total += cap * (size_t) hp.kv_dim * sizeof(__half) * 2;
        }
        StateCheckpoint cp;
        cp.tokens = tokens;
        cp.data.resize(total);
        size_t off = 0;
        cudaStream_t copy_stream = st;
        auto pull = [&](const void* src, size_t n) {
            CK(cudaMemcpyAsync(cp.data.data() + off, src, n, cudaMemcpyDeviceToHost, copy_stream));
            off += n;
        };
        for (int il = 0; il < hp.n_layer; ++il) {
            Engine& lane = runtime_for(il);
            DeviceScope owner(lane.device);
            copy_stream = lane.st;
            pull(lane.conv_hist[(size_t) il], conv_bytes);
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            if (kv_q8) {
                pull(lane.kq[(size_t) il], cap * (size_t) hp.kv_dim);
                pull(lane.vq[(size_t) il], cap * (size_t) hp.kv_dim);
                pull(lane.ks[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
                pull(lane.vs[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
            } else {
                pull(lane.kc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
                pull(lane.vc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
            }
        }
        if (dspark && role != Role::Core) {
            DeviceScope owner(device);
            cp.dspark_data.resize(dspark->checkpoint_bytes());
            dspark->save_checkpoint(cp.dspark_data.data(), st);
        }
        { DeviceScope owner(device); CK(cudaStreamSynchronize(st)); }
        if (split_peer) { DeviceScope owner(split_peer->device); CK(cudaStreamSynchronize(split_peer->st)); }
        return cp;
    }

    void restore_checkpoint(const StateCheckpoint& cp) {
        const auto& hp = m->hp;
        const size_t conv_bytes = (size_t) hp.n_sconv_channels() * 3 * sizeof(float);
        size_t off = 0;
        cudaStream_t copy_stream = st;
        auto push = [&](void* dst, size_t n) {
            CK(cudaMemcpyAsync(dst, cp.data.data() + off, n, cudaMemcpyHostToDevice, copy_stream));
            off += n;
        };
        for (int il = 0; il < hp.n_layer; ++il) {
            Engine& lane = runtime_for(il);
            DeviceScope owner(lane.device);
            copy_stream = lane.st;
            push(lane.conv_hist[(size_t) il], conv_bytes);
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            if (kv_q8) {
                push(lane.kq[(size_t) il], cap * (size_t) hp.kv_dim);
                push(lane.vq[(size_t) il], cap * (size_t) hp.kv_dim);
                push(lane.ks[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
                push(lane.vs[(size_t) il], cap * (size_t) hp.n_head_kv * sizeof(__half));
            } else {
                push(lane.kc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
                push(lane.vc[(size_t) il], cap * (size_t) hp.kv_dim * sizeof(__half));
            }
        }
        pos = (int64_t) cp.tokens.size();
        if (dspark && role != Role::Core && !cp.dspark_data.empty()) {
            DeviceScope owner(device);
            dspark->restore_checkpoint(cp.dspark_data.data(), (int) pos, st);
        }
        if (split_peer) {
            split_peer->pos = pos;
            { DeviceScope owner(device); CK(cudaStreamSynchronize(st)); }
            { DeviceScope owner(split_peer->device); CK(cudaStreamSynchronize(split_peer->st)); }
        }
    }

    ~Engine() {
        DeviceScope owner(device);
        if (replica) {
            cudaStreamSynchronize(st);
            print_replica_stats("FINAL");
            if (replica->input_ready) cudaEventDestroy(replica->input_ready);
            cudaFree(replica->frequencies); cudaFree(replica->hits);
            cudaFree(replica->local_miss); cudaFree(replica->local_parts);
            replica.reset();
        }
        if (core_graph) cudaGraphExecDestroy(core_graph);
        if (capture_core_graph) cudaGraphExecDestroy(capture_core_graph);
        if (verify_core_graph) cudaGraphExecDestroy(verify_core_graph);
        for (auto graph : outer_graph) if (graph) cudaGraphExecDestroy(graph);
        for (auto g : layer_pre_graph) if (g) cudaGraphExecDestroy(g);
        for (auto* p : kc) cudaFree(p);
        for (auto* p : vc) cudaFree(p);
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        for (auto* p : conv_hist) cudaFree(p);
        cudaFree(swa_bak); cudaFree(d_conv_trace);
        for (auto* p : d_tgt_feat) cudaFree(p);
        cudaFreeHost(split_features);
        for (auto* p : norm_blk_buf) cudaFree(p);
        cudaFree(prelude_cur_buf);
        cudaFree(norm_cur);
        cudaFree(cur); cudaFree(ffn_inp);
        cudaFree(in_q); cudaFree(in_k); cudaFree(in_v); cudaFree(in_ffn);
        cudaFree(msg0); cudaFree(msg1); cudaFree(msg2); cudaFree(msg3);
        cudaFree(ctrl2N); cudaFree(ctrl64); cudaFree(ctrl_g); cudaFree(ctrl_f);
        cudaFree(qkv_ctrl2N); cudaFree(qkv_ctrl64);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(R); cudaFree(qgr_t1);
        cudaFree(K_conv); cudaFree(V_conv); cudaFree(dyn_k); cudaFree(dyn_v);
        cudaFree(sink); cudaFree(rel_tab); cudaFree(att); cudaFree(att_gate); cudaFree(wo);
        cudaFree(attn_split);
        cudaFree(router_logits); cudaFree(ewts); cudaFree(eparts); cudaFree(moe_raw);
        cudaFree(logits); cudaFree(act_bf16);
        cudaFree(xq); cudaFree(xq_k); cudaFree(xq_v); cudaFree(escr);
        cudaFree(ids); cudaFree(grp_start); cudaFree(n_groups); cudaFree(ent_dst); cudaFree(ent_tok); cudaFree(grp_ptr);
        cudaFree(tok_d); cudaFree(pos0_d); cudaFree(pick_d);
        cudaFree(d_miss); cudaFree(ecpu);
        cudaFreeHost(h_tok_buf);
        cudaFreeHost(h_ids); cudaFreeHost(h_h); cudaFreeHost(cpu_out); cudaFreeHost(h_miss);
        cudaFreeHost(split_handoff);
        std::free(acts);
        if (ev_route) cudaEventDestroy(ev_route);
        for (int b = 0; b < 2; ++b) {
            if (pp_stage[b]) cudaFree(pp_stage[b]);
            if (ev_pp_copy[b]) cudaEventDestroy(ev_pp_copy[b]);
            if (ev_pp_used[b]) cudaEventDestroy(ev_pp_used[b]);
        }
        if (ev_rebal) cudaEventDestroy(ev_rebal);
        if (cst) cudaStreamDestroy(cst);
        // Gemm owns CUDA allocations/handles too: destroy it while its owner's device is current.
        gemm.reset();
        if (blas) cublasDestroy(blas);
        if (st) cudaStreamDestroy(st);
    }

    bool init(Model& model, int max_ctx, int max_t, bool use_kv_q8, std::string& err,
              int max_verify_t = 1, bool draft_quant = false, int draft_ff = 0) {
        m = &model;
        if (device < 0) CK(cudaGetDevice(&device));
        DeviceScope owner(device);
        ctx = max_ctx;
        verify_max_t = std::max(1, max_verify_t);
        tmax = std::max(max_t, verify_max_t);
        kv_q8 = use_kv_q8;
        const auto& hp = m->hp;
        const int64_t N = hp.n_embd;
        const int64_t projection_in = std::max<int64_t>(2 * N, draft_ff);

        CK(cudaStreamCreate(&st));
        CKB(cublasCreate(&blas));
        CKB(cublasSetStream(blas, st));
        gemm = std::make_unique<strata::prefill::Gemm>();
        if (!gemm->init(st, m->has_quant_dense || draft_quant ? (size_t) (8 * 1024 * 1024) : 4096, err)) return false;

        // Allocate KV caches & recurrent history per virtual layer
        layer_pre_graph.assign((size_t) hp.n_layer, nullptr);
        pp_layer_cnt.assign((size_t) hp.n_layer, 0);
        kc.resize((size_t) hp.n_layer, nullptr);
        vc.resize((size_t) hp.n_layer, nullptr);
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer);
        conv_hist.resize((size_t) hp.n_layer);
        layer_conv_floats = (size_t) hp.n_sconv_channels() * 3;
        trace_slot.assign((size_t) hp.n_layer, -1);
        for (int il = 0; il < hp.n_layer; ++il) {
            // All T rows are stored before attention. Keep the union needed by the earliest
            // and latest query: window + T - 1 positions, while attention still uses window.
            const int cap = hp.is_swa(il) ? (int) hp.n_swa + tmax - 1 : ctx;
            kv_cap[(size_t) il] = cap;
            if (!owns_virtual(il)) continue;
            trace_slot[(size_t) il] = (int) (local_conv_floats / layer_conv_floats);
            local_conv_floats += layer_conv_floats;
            if (kv_q8) {
                CK(cudaMalloc(&kq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
                CK(cudaMalloc(&vq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
                CK(cudaMalloc(&ks[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
                CK(cudaMalloc(&vs[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
            } else {
                CK(cudaMalloc(&kc[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(__half)));
                CK(cudaMalloc(&vc[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(__half)));
            }
            CK(cudaMalloc(&conv_hist[(size_t) il], (size_t) hp.n_sconv_channels() * 3 * sizeof(float)));
        }

        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        for (int b = 0; b < 9; ++b) {
            d_f32(&norm_blk_buf[b], (size_t) tmax * N);
        }
        d_f32(&prelude_cur_buf, (size_t) tmax * N);
        d_f32(&norm_cur, (size_t) tmax * N);
        d_f32(&cur, (size_t) tmax * N);
        d_f32(&ffn_inp, (size_t) tmax * N);
        d_f32(&in_q, (size_t) tmax * N);
        d_f32(&in_k, (size_t) tmax * N);
        d_f32(&in_v, (size_t) tmax * N);
        d_f32(&in_ffn, (size_t) tmax * N);
        d_f32(&msg0, (size_t) tmax * N);
        d_f32(&msg1, (size_t) tmax * N);
        d_f32(&msg2, (size_t) tmax * N);
        d_f32(&msg3, (size_t) tmax * N);
        d_f32(&ctrl2N, (size_t) tmax * 2 * N);
        d_f32(&ctrl64, (size_t) tmax * hp.loop_gate_rank);
        d_f32(&ctrl_g, (size_t) tmax * N);
        d_f32(&ctrl_f, (size_t) tmax * N);
        if (batch_qkv_control_t1) {
            d_f32(&qkv_ctrl2N, (size_t) 3 * 2 * N);
            d_f32(&qkv_ctrl64, (size_t) 3 * 64);
        }

        d_f32(&Q, (size_t) tmax * hp.q_dim);
        d_f32(&K, (size_t) tmax * hp.kv_dim);
        d_f32(&V, (size_t) tmax * hp.kv_dim);
        d_f32(&R, (size_t) tmax * hp.n_rel * hp.n_head);
        d_f32(&qgr_t1, (size_t) (2 * hp.q_dim + hp.n_rel * hp.n_head + 2 * hp.kv_dim + 2 * hp.n_conv * hp.n_head_kv));
        d_f32(&K_conv, (size_t) tmax * hp.kv_dim);
        d_f32(&V_conv, (size_t) tmax * hp.kv_dim);
        d_f32(&dyn_k, (size_t) tmax * hp.n_conv * hp.n_head_kv);
        d_f32(&dyn_v, (size_t) tmax * hp.n_conv * hp.n_head_kv);
        d_f32(&sink, (size_t) tmax * hp.n_head);
        d_f32(&rel_tab, (size_t) tmax * hp.n_head * hp.n_rel_ext);
        d_f32(&att, (size_t) tmax * hp.q_dim);
        d_f32(&att_gate, (size_t) tmax * hp.q_dim);
        d_f32(&wo, (size_t) tmax * N);
        d_f32(&attn_split, (size_t) kAttnSplit * hp.n_head * 130);

        const int64_t max_ent = (int64_t) tmax * hp.n_expert_used;
        d_f32(&router_logits, (size_t) tmax * hp.n_expert);
        d_f32(&ewts, (size_t) max_ent);
        d_f32(&eparts, (size_t) max_ent * N);
        d_f32(&moe_raw, (size_t) tmax * N);
        d_f32(&logits, (size_t) verify_max_t * hp.vocab_unpadded);

        CK(cudaMallocHost(&h_tok_buf, (size_t) tmax * N * sizeof(float)));
        CK(cudaMalloc(&act_bf16, (size_t) tmax * projection_in * sizeof(uint16_t)));

        CK(cudaMalloc(&xq, (size_t) tmax * (projection_in / 32) * 36));
        CK(cudaMalloc(&xq_k, (size_t) tmax * ((2 * N) / 32) * 36));
        CK(cudaMalloc(&xq_v, (size_t) tmax * ((2 * N) / 32) * 36));
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes(max_ent, hp.n_ff)));
        CK(cudaMalloc(&ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_start, (size_t) (hp.n_expert + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&n_groups, sizeof(int32_t)));
        CK(cudaMalloc(&ent_dst, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ent_tok, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaMalloc(&grp_ptr, (size_t) hp.n_expert * sizeof(unsigned long long)));
        CK(cudaMalloc(&tok_d, (size_t) tmax * sizeof(int32_t)));
        CK(cudaMalloc(&pos0_d, sizeof(int32_t)));
        CK(cudaMalloc(&pick_d, (size_t) tmax * sizeof(int32_t)));

        CK(cudaMallocHost(&h_ids, (size_t) max_ent * sizeof(int32_t)));
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_rebal, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CK(cudaEventCreateWithFlags(&ev_pp_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_pp_used[b], cudaEventDisableTiming));
        }
        if (const char* v = std::getenv("STRATA_PP_STREAM_T")) pp_stream_t = std::max(1, std::atoi(v));
        reset_state();
        return true;
    }

    void init_cpu_bufs(int cpu_t) {
        DeviceScope owner(device);
        const int64_t N = m->hp.n_embd;
        const int64_t cpu_ent = (int64_t) cpu_t * m->hp.n_expert_used;
        cpu_rows = cpu_t;
        if (const char* v = std::getenv("STRATA_CPU_ROWS_TEST")) cpu_rows = std::max(1, std::min(cpu_t, std::atoi(v)));
        CK(cudaMallocHost(&h_h, (size_t) cpu_t * N * sizeof(float)));
        CK(cudaMallocHost(&cpu_out, (size_t) cpu_ent * N * sizeof(float)));
        CK(cudaMallocHost(&h_miss, (size_t) cpu_ent * sizeof(int32_t)));
        CK(cudaMalloc(&d_miss, (size_t) cpu_ent * sizeof(int32_t)));
        CK(cudaMalloc(&ecpu, (size_t) cpu_ent * N * sizeof(float)));
#if defined(STRATA_INKLING2_CPU_POOL)
        acts = (uint8_t*) std::malloc((size_t) cpu_t * strata::kernels::cpu::kNativeActBytes);
#endif
    }

    void reset_state() {
        DeviceScope owner(device);
        if (replica) refresh_replica_hot();
        pos = 0;
        for (int il = 0; il < m->hp.n_layer; ++il) {
            if (!owns_virtual(il)) continue;
            CK(cudaMemsetAsync(conv_hist[(size_t) il], 0, (size_t) m->hp.n_sconv_channels() * 3 * sizeof(float), st));
        }
        if (role == Role::Outer && split_peer) split_peer->reset_state();
        if (dspark && role != Role::Core) dspark->reset(st);
    }

    void enable_dspark(DsparkEngine* draft) {
        DeviceScope owner(device);
        dspark = draft;
        for (auto& p : d_tgt_feat) CK(cudaMalloc(&p, (size_t) tmax * m->hp.n_embd * sizeof(float)));
        CK(cudaMalloc(&d_conv_trace, (size_t) verify_max_t * local_conv_floats * sizeof(float)));
        for (int il = 0; il < m->hp.n_layer; ++il) if (owns_virtual(il) && m->hp.is_swa(il)) {
            const int s = swa_ptrs.n_swa_layers++;
            swa_ptrs.kq[s] = kq[(size_t) il]; swa_ptrs.vq[s] = vq[(size_t) il];
            swa_ptrs.ks[s] = ks[(size_t) il]; swa_ptrs.vs[s] = vs[(size_t) il];
            swa_ptrs.kc[s] = kc[(size_t) il]; swa_ptrs.vc[s] = vc[(size_t) il];
            swa_ring_cap = kv_cap[(size_t) il];
        }
        const size_t slot_bytes = (size_t) m->hp.kv_dim * 2 * (kv_q8 ? 1 : sizeof(__half)) +
                                 (kv_q8 ? (size_t) m->hp.n_head_kv * 2 * sizeof(__half) : 0);
        if (swa_ptrs.n_swa_layers > 0)
            CK(cudaMalloc(&swa_bak, (size_t) std::max(1, verify_max_t - 1) * swa_ptrs.n_swa_layers * slot_bytes));
        if (role == Role::Outer)
            CK(cudaHostAlloc(&split_features, (size_t) 2 * tmax * m->hp.n_embd * sizeof(float), cudaHostAllocPortable));
    }

    void backup_verify(int K) {
        DeviceScope owner(device);
        if (K <= 0 || !swa_bak) return;
        if (kv_q8)
            k_swa_q8_bak<<<dim3((unsigned) K, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                swa_ptrs, swa_bak, (int) m->hp.kv_dim, (int) m->hp.n_head_kv, swa_ring_cap, (int) pos + 1, 0, K, 0);
        else
            k_swa_f16_bak<<<dim3((unsigned) K, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                swa_ptrs, swa_bak, (int) m->hp.kv_dim, swa_ring_cap, (int) pos + 1, 0, K, 0);
    }

    void commit_verify(int accepted, int K, int pos_pre) {
        DeviceScope owner(device);
        if (accepted < K) {
            for (int il = 0; il < m->hp.n_layer; ++il) if (owns_virtual(il))
                CK(cudaMemcpyAsync(conv_hist[(size_t) il], d_conv_trace + (size_t) accepted * local_conv_floats +
                                   (size_t) trace_slot[(size_t) il] * layer_conv_floats,
                                   layer_conv_floats * sizeof(float), cudaMemcpyDeviceToDevice, st));
            const int n = K - accepted;
            if (swa_ptrs.n_swa_layers > 0) {
                if (kv_q8)
                    k_swa_q8_bak<<<dim3((unsigned) n, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                        swa_ptrs, swa_bak, (int) m->hp.kv_dim, (int) m->hp.n_head_kv, swa_ring_cap, pos_pre + 1, accepted, n, 1);
                else
                    k_swa_f16_bak<<<dim3((unsigned) n, (unsigned) swa_ptrs.n_swa_layers), 256, 0, st>>>(
                        swa_ptrs, swa_bak, (int) m->hp.kv_dim, swa_ring_cap, pos_pre + 1, accepted, n, 1);
            }
        }
        pos = pos_pre + 1 + accepted;
        if (role == Role::Outer && split_peer) split_peer->commit_verify(accepted, K, pos_pre);
    }

    void prep_q8(const float* X, void* q8_buf, int T, int64_t K_in) {
        if (T <= 8 && (K_in % 32) == 0) {
            K::native_quantize_q8_1(X, q8_buf, (int) K_in, T, st);
        }
    }

    void project_preq(const float* X, const void* q8_ready, const DenseMat& W, float* Y, int T, int64_t N_out, int64_t K_in) {
        if (W.type == 0) {
            project_f32(X, (const float*) W.d, Y, T, N_out, K_in);
            return;
        }
        if (W.type == 1) {
            const int64_t tot = (int64_t) T * K_in;
            k_f32_to_f16<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(X, act_bf16, tot);
            gemm->f16(act_bf16, (const uint16_t*) W.d, Y, T, N_out, K_in);
            return;
        }
        if (W.type == 30) {
            const uint16_t* wp = (const uint16_t*) W.d;
            if (T == 1) {
                K::bf16_gemv_fp32_mmvf(X, wp, Y, K_in, N_out, st);
            } else {
                const int64_t tot = (int64_t) T * K_in;
                k_f32_to_bf16<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(X, act_bf16, tot);
                gemm->bf16(act_bf16, wp, Y, T, N_out, K_in);
            }
            return;
        }
        const auto* ti = strata::quant::type_info(W.type);
        const int blck = ti ? ti->blck : 256;
        if (T <= 8 && (K_in % blck) == 0) {
            if (!q8_ready) K::native_quantize_q8_1(X, xq, (int) K_in, T, st);
            K::native_mmvq(W.type, W.d, q8_ready ? q8_ready : xq, Y, (int) K_in, (int) N_out, T, st);
            return;
        }
        const int64_t tot = (int64_t) T * K_in;
        k_f32_to_f16<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(X, act_bf16, tot);
        gemm->native(act_bf16, W.type, W.d, Y, T, N_out, K_in);
    }

    void project(const float* X, const DenseMat& W, float* Y, int T, int64_t N_out, int64_t K_in) {
        project_preq(X, nullptr, W, Y, T, N_out, K_in);
    }

    void project_f32(const float* X, const float* W, float* Y, int T, int64_t N_out, int64_t K_in) {
        if (T == 1 && (K_in % 4) == 0) {
            if (N_out <= 128) {
                k_gemv_f32_cta<<<(unsigned) N_out, 128, 0, st>>>(W, X, Y, (int) K_in, (int) N_out);
            } else {
                k_gemv_f32<<<(unsigned) ((N_out + 7) / 8), 256, 0, st>>>(W, X, Y, (int) K_in, (int) N_out);
            }
            return;
        }
        const float alpha = 1.0f, beta = 0.0f;
        CKB(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) N_out, T, (int) K_in, &alpha, W, (int) K_in, X, (int) K_in, &beta, Y, (int) N_out));
    }

    void run_mix_normed(const std::vector<float*>& n_blocks, const float* c, bool c_already_normed, const float* query, float* const* outs, int n_q, int T, int swa_bcast) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int B = (int) n_blocks.size();
        const int D = B + 1;
        if (!c_already_normed) {
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(c, nullptr, norm_cur, N, hp.rms_eps);
        }
        NormMixPtrs np{};
        for (int i = 0; i < B; ++i) np.p[i] = n_blocks[(size_t) (B - 1 - i)];
        np.p[B] = c_already_normed ? c : norm_cur;
        MixOuts mo{};
        for (int i = 0; i < (swa_bcast ? 3 : n_q); ++i) mo.p[i] = outs[i];
        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
        k_mix_normed<<<dim3((unsigned) n_q, (unsigned) T), kThreads, 0, st>>>(np, D, query, mo, n_q, N, inv_sqrt_n, swa_bcast);
    }

    void run_control(const float* c, const float* msg, const PhysLayer& L, int mod, float* out, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        k_control_prep<<<dim3(2, (unsigned) T), kThreads, 0, st>>>(c, msg, ctrl2N, N, hp.rms_eps);
        project(ctrl2N, L.loop_down[mod], ctrl64, T, hp.loop_gate_rank, 2 * N);
        if (T == 1 && hp.loop_gate_rank == 64 && L.loop_gain[mod].type == 0 && L.loop_forget[mod].type == 0) {
            k_control_up_combine_t1<<<(unsigned) ((N + 31) / 32), 256, 0, st>>>(
                c, msg, (const float*) L.loop_gain[mod].d, (const float*) L.loop_forget[mod].d,
                ctrl64, out, (int) hp.loop_gate_rank, N);
            return;
        }
        project(ctrl64, L.loop_gain[mod], ctrl_g, T, N, hp.loop_gate_rank);
        project(ctrl64, L.loop_forget[mod], ctrl_f, T, N, hp.loop_gate_rank);
        const int64_t tot = (int64_t) T * N;
        k_control_combine<<<(unsigned) ((tot + 255) / 256), 256, 0, st>>>(c, msg, ctrl_g, ctrl_f, out, tot);
    }

    void run_controls_qkv(const float* q, const float* k, const float* v, const PhysLayer& L, int T) {
        bool f32 = batch_qkv_control_t1 && T == 1 && m->hp.loop_gate_rank == 64 && (m->hp.n_embd % 4) == 0;
        if (f32) {
            for (int i = 0; i < 3; ++i)
                f32 = f32 && L.loop_down[i].type == 0 && L.loop_gain[i].type == 0 && L.loop_forget[i].type == 0;
        }
        if (f32) {
            const int N = (int) m->hp.n_embd;
            QkvControlT1 p{};
            p.cur[0] = q; p.cur[1] = k; p.cur[2] = v;
            p.msg[0] = msg0; p.msg[1] = msg1; p.msg[2] = msg2;
            p.out[0] = in_q; p.out[1] = in_k; p.out[2] = in_v;
            for (int i = 0; i < 3; ++i) {
                p.down[i] = (const float*) L.loop_down[i].d;
                p.gain[i] = (const float*) L.loop_gain[i].d;
                p.forget[i] = (const float*) L.loop_forget[i].d;
            }
            k_control_prep_qkv_t1<<<dim3(2, 3), kThreads, 0, st>>>(p, qkv_ctrl2N, N, m->hp.rms_eps);
            k_control_down_qkv_t1<<<dim3(64, 3), 128, 0, st>>>(p, qkv_ctrl2N, qkv_ctrl64, N);
            k_control_up_combine_qkv_t1<<<dim3((unsigned) ((N + 31) / 32), 3), 256, 0, st>>>(p, qkv_ctrl64, N);
            return;
        }
        run_control(q, msg0, L, 0, in_q, T);
        run_control(k, msg1, L, 1, in_k, T);
        run_control(v, msg2, L, 2, in_v, T);
    }

    void pp_kick_next() {
        if (pp_kick_idx >= pp_tasks.size()) return;
        const PpTask& tk = pp_tasks[pp_kick_idx];
        const int b = (int) (pp_kick_idx & 1u);
        const PhysLayer& L = m->layers[(size_t) tk.ip];
        CK(cudaStreamWaitEvent(cst, ev_pp_used[b], 0));
        for (const PpRun& r : tk.runs) {
            const size_t nbytes = (size_t) (r.cnt - 1) * L.stride + L.L.bytes;
            CK(cudaMemcpyAsync(pp_stage[b] + (size_t) r.dst_off * L.stride,
                               L.hblobs + (size_t) r.e0 * L.stride,
                               nbytes, cudaMemcpyHostToDevice, cst));
        }
        CK(cudaEventRecord(ev_pp_copy[b], cst));
        ++pp_kick_idx;
    }

    void pp_begin(int core_end_virt, int core_all_gpu_flag) {
        const auto& hp = m->hp;
        const int ne = (int) hp.n_expert;
        pp_tasks.clear();
        std::fill(pp_layer_cnt.begin(), pp_layer_cnt.end(), 0);
        pp_take_idx = 0;
        pp_kick_idx = 0;
        if (!pp_stage[0] || pp_stage_bytes == 0) return;
        (void) core_end_virt;
        (void) core_all_gpu_flag;
        int missing[128];
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!owns_virtual(il)) continue;
            const int ip = hp.loop_layer(il);
            const PhysLayer& L = m->layers[(size_t) ip];
            if (L.n_slots >= ne || L.gu_type == 30 || !L.hblobs) continue;
            const int cap = (int) std::min<size_t>((size_t) ne, pp_stage_bytes / L.stride);
            if (cap <= 0) continue;
            int n_miss = 0;
            for (int x = 0; x < ne; ++x) {
                if (L.slot_of[(size_t) x] < 0) missing[n_miss++] = x;
            }
            for (int k0 = 0; k0 < n_miss; k0 += cap) {
                const int k1 = std::min(n_miss, k0 + cap);
                PpTask tk;
                tk.ip = ip;
                tk.n_staged = k1 - k0;
                std::fill(std::begin(tk.map.pos), std::end(tk.map.pos), int16_t(-1));
                for (int i = k0; i < k1; ++i) {
                    const int x = missing[i];
                    const int dst = i - k0;
                    tk.map.pos[x] = (int16_t) dst;
                    if (!tk.runs.empty() && (int) tk.runs.back().e0 + (int) tk.runs.back().cnt == x) {
                        ++tk.runs.back().cnt;
                    } else {
                        tk.runs.push_back(PpRun{(int16_t) x, 1, (int16_t) dst});
                    }
                }
                pp_tasks.push_back(std::move(tk));
                ++pp_layer_cnt[(size_t) il];
            }
        }
        for (int b = 0; b < 2; ++b) CK(cudaEventRecord(ev_pp_used[b], st));
        for (int k = 0; k < 2 && pp_kick_idx < pp_tasks.size(); ++k) pp_kick_next();
    }

    // Only called at request reset, after the previous request has completed.
    // Scores contain actual routed entries, including all three core visits.
    void refresh_replica_hot() {
        auto& r = *replica;
        CK(cudaStreamSynchronize(st));
        { DeviceScope remote(r.device); CK(cudaStreamSynchronize(r.stream)); }
        collect_replica_stats(true);
        const int ne = (int) m->hp.n_expert;
        std::vector<uint8_t> blob;
        int loaded = 0;
        for (size_t ip = 0; ip < m->layers.size(); ++ip) {
            auto& L = m->layers[ip];
            const auto order = replica_hot_order(L.score, L.n_slots);
            std::fill(L.slot_of.begin(), L.slot_of.end(), -1);
            std::vector<int32_t> cold((size_t) ne);
            for (int e = 0; e < ne; ++e) cold[e] = e;
            blob.resize(L.L.bytes);
            for (int slot = 0; slot < (int) order.size(); ++slot) {
                const int e = order[slot];
                build_expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.d_blobs + (size_t) slot * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
                L.slot_of[e] = slot;
                cold[e] = -1;
                ++loaded;
            }
            CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice));
            for (int e = 0; e < ne; ++e) if ((L.slot_of[e] >= 0) + (cold[e] >= 0) != 1) {
                std::fprintf(stderr, "strata-inkling2: invalid replica complement map\n"); std::exit(1);
            }
            r.cold_of[ip] = cold;
            { DeviceScope remote(r.device);
              CK(cudaMemcpy(r.maps[ip], cold.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice)); }
        }
        std::fprintf(stderr, "REPLICA {\"event\":\"RESET\",\"hot_loaded\":%d,\"source\":\"observed_router_counts\",\"update\":\"request_reset\"}\n", loaded);
    }

    void run_replica_moe_async(PhysLayer& L, int il, const float* x_norm, int T) {
        auto& r = *replica;
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd, top = (int) hp.n_expert_used, ne = (int) hp.n_expert;
        const int entries = T * top, ip = hp.loop_layer(il);
        const size_t qbytes = (size_t) T * (N / 32) * 36, pbytes = (size_t) entries * N * sizeof(float);
        K::quantize_q8_1_rows(x_norm, T, N, xq, st);
        k_replica_route_stats<<<(entries+255)/256,256,0,st>>>(ids, L.d_slot_of, r.local_miss,
            r.frequencies + (size_t) ip * ne, r.hits, entries, ne);
        CK(cudaMemcpyAsync(r.host_ids, ids, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(r.host_xq, xq, qbytes, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(r.input_ready, st));
        // Record each generation before issuing its cross-device wait. The primary tail
        // consumes the return slot before the next input_ready lets remote reuse it.
        if (L.n_slots) {
            k_group_experts<<<1,128,0,st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, entries, ne, L.n_slots,
                grp_ptr, grp_start, n_groups, ent_dst, ent_tok, top);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                std::min(entries, ne), entries, xq, escr, eparts, st);
        }
        { DeviceScope remote(r.device);
          CK(cudaStreamWaitEvent(r.stream, r.input_ready, 0));
          CK(cudaMemcpyAsync(r.ids, r.host_ids, (size_t) entries * sizeof(int32_t), cudaMemcpyHostToDevice, r.stream));
          CK(cudaMemcpyAsync(r.xq, r.host_xq, qbytes, cudaMemcpyHostToDevice, r.stream));
          k_group_experts<<<1,128,0,r.stream>>>(r.ids, r.maps[ip], r.blobs[ip], L.stride, entries, ne, ne,
              r.ptrs, r.starts, r.groups, r.dst, r.tok, top);
          K::native_expert_grouped(L.L, r.ptrs, r.starts, r.groups, r.dst, r.tok,
              std::min(entries, ne), entries, r.xq, r.scratch, r.parts, r.stream);
          CK(cudaMemcpyAsync(r.host_parts, r.parts, pbytes, cudaMemcpyDeviceToHost, r.stream));
          CK(cudaEventRecord(r.remote_done, r.stream)); }
        CK(cudaStreamWaitEvent(st, r.remote_done, 0));
        CK(cudaMemcpyAsync(r.local_parts, r.host_parts, pbytes, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<(unsigned) entries,kThreads,0,st>>>(eparts, r.local_parts, r.local_miss, N);
        k_combine_experts<<<dim3((unsigned)((N+255)/256),(unsigned)T),256,0,st>>>(eparts, ewts, moe_raw, N, top);
        hy_gpu += entries;
        r.relay_bytes += (int64_t) 2 * (qbytes + (size_t) entries * sizeof(int32_t) + pbytes);
    }

    void run_replica_moe(PhysLayer& L, int il, const float* x_norm, int T) {
        auto& r = *replica;
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd, top = (int) hp.n_expert_used, ne = (int) hp.n_expert;
        const int entries = T * top;
        const size_t qbytes = (size_t) T * (N / 32) * 36;
        const size_t pbytes = (size_t) entries * N * sizeof(float);
        K::quantize_q8_1_rows(x_norm, T, N, xq, st);
        CK(cudaMemcpyAsync(r.host_ids, ids, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(r.host_xq, xq, qbytes, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st)); // portable pinned host relay, never peer/UVA
        int misses = 0;
        const int ip = hp.loop_layer(il);
        for (int e = 0; e < entries; ++e) {
            const int x = r.host_ids[e];
            if (x < 0 || x >= ne) { std::fprintf(stderr, "strata-inkling2: invalid replica router ID\n"); std::exit(1); }
            const int owners = (L.slot_of[x] >= 0) + (r.cold_of[ip][x] >= 0);
            r.duplicate_entries += owners > 1;
            r.missing_entries += owners == 0;
            if (owners != 1) { std::fprintf(stderr, "strata-inkling2: replica ownership partition failed\n"); std::exit(1); }
            L.score[x] += 1.0;
            r.host_miss[e] = L.slot_of[x] < 0;
            misses += r.host_miss[e];
        }
        hy_gpu += entries;
        r.local_hits += entries - misses;
        r.remote_hits += misses;
        const auto join_start = Clock::now();
        // Launch cold work first; host switches context while this stream computes.
        if (misses) {
            DeviceScope remote(r.device);
            CK(cudaMemcpyAsync(r.ids, r.host_ids, (size_t) entries * sizeof(int32_t), cudaMemcpyHostToDevice, r.stream));
            CK(cudaMemcpyAsync(r.xq, r.host_xq, qbytes, cudaMemcpyHostToDevice, r.stream));
            k_group_experts<<<1,128,0,r.stream>>>(r.ids, r.maps[ip], r.blobs[ip], L.stride, entries, ne, ne,
                r.ptrs, r.starts, r.groups, r.dst, r.tok, top);
            K::native_expert_grouped(L.L, r.ptrs, r.starts, r.groups, r.dst, r.tok,
                std::min(entries, ne), entries, r.xq, r.scratch, r.parts, r.stream);
            CK(cudaMemcpyAsync(r.host_parts, r.parts, pbytes, cudaMemcpyDeviceToHost, r.stream));
        }
        if (entries > misses) {
            k_group_experts<<<1,128,0,st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, entries, ne, L.n_slots,
                grp_ptr, grp_start, n_groups, ent_dst, ent_tok, top);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                std::min(entries, ne), entries, xq, escr, eparts, st);
        }
        if (misses) {
            { DeviceScope remote(r.device); CK(cudaStreamSynchronize(r.stream)); }
            CK(cudaMemcpyAsync(r.local_parts, r.host_parts, pbytes, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(r.local_miss, r.host_miss, (size_t) entries * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            k_merge_cpu<<<(unsigned) entries,kThreads,0,st>>>(eparts, r.local_parts, r.local_miss, N);
            r.relay_bytes += (int64_t) 2 * (qbytes + (size_t) entries * sizeof(int32_t) + pbytes);
        }
        // Original top-k entry order and combine arithmetic are preserved.
        k_combine_experts<<<dim3((unsigned)((N+255)/256),(unsigned)T),256,0,st>>>(eparts, ewts, moe_raw, N, top);
        CK(cudaStreamSynchronize(st)); // staging reused on the next physical visit
        r.join_ms += ms_since(join_start);
    }

    void run_moe(PhysLayer& L, int il, const float* x_norm, int T) {
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd, Kp = (int) hp.n_expert_used;
        const int n_ent = T * Kp;
        const auto tp_moe_r0 = Clock::now();
        project_f32(x_norm, L.ffn_gate_inp, router_logits, T, hp.n_expert, N);
        k_route_inkling2<<<(unsigned) T, 32, 0, st>>>(router_logits, L.exp_probs_b, ids, ewts, (int) hp.n_expert, Kp, hp.expert_weights_scale);
        if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_route += ms_since(tp_moe_r0); }

        if (replica) {
            if (replica->async) run_replica_moe_async(L, il, x_norm, T);
            else run_replica_moe(L, il, x_norm, T);
            return;
        }
        const bool has_gpu = L.n_slots > 0;
        const bool need_cpu = L.n_slots < hp.n_expert;
        const bool is_q5 = (L.L.gu_type == 13);

        if (need_cpu && has_pending_rebal) {
            CK(cudaStreamWaitEvent(st, ev_rebal, 0));
            has_pending_rebal = false;
        }

        if (pp_streaming && need_cpu && L.gu_type != 30 && L.hblobs && pp_stage_bytes >= L.stride) {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
            K::quantize_q8_1_rows(x_norm, T, N, xq, st);
            if (has_gpu) {
                k_group_experts<<<1, 128, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, (int) hp.n_expert, L.n_slots,
                                                   grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                         std::min<int64_t>(n_ent, hp.n_expert), n_ent, xq, escr, eparts, st);
            }
            const int ne = (int) hp.n_expert;
            const int n_chunks = pp_layer_cnt[(size_t) il];
            for (int ci = 0; ci < n_chunks; ++ci) {
                if (pp_take_idx >= pp_tasks.size()) {
                    std::fprintf(stderr, "strata-inkling2: pp stream underflow\n");
                    std::exit(1);
                }
                const PpTask& tk = pp_tasks[pp_take_idx];
                const int b = (int) (pp_take_idx & 1u);
                CK(cudaStreamWaitEvent(st, ev_pp_copy[b], 0));
                k_group_experts_miss_map<<<1, 128, 0, st>>>(ids, pp_stage[b], L.stride, n_ent, ne, tk.map,
                                                            grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
                K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                         std::min<int64_t>(n_ent, (int64_t) tk.n_staged), n_ent, xq, escr, eparts, st);
                CK(cudaEventRecord(ev_pp_used[b], st));
                ++pp_take_idx;
                pp_kick_next();
            }
            k_combine_experts<<<dim3((unsigned) ((N + 255) / 256), (unsigned) T), 256, 0, st>>>(eparts, ewts, moe_raw, N, Kp);
            if (prof) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }
            CK(cudaEventSynchronize(ev_route));
            int n_miss = 0;
            for (int e = 0; e < n_ent; ++e) {
                const int x = h_ids[e];
                L.score[(size_t) x] += 1.0;
                if (L.slot_of[(size_t) x] < 0) ++n_miss;
            }
            hy_cpu += n_miss;
            hy_gpu += n_ent - n_miss;
            return;
        }

        if (need_cpu) {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, x_norm, (size_t) std::min(T, cpu_rows > 0 ? cpu_rows : T) * N * sizeof(float), cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
        }

        const auto tp_mg0 = Clock::now();
        if (has_gpu) {
            k_group_experts<<<1, 128, 0, st>>>(ids, L.d_slot_of, L.d_blobs, L.stride, n_ent, (int) hp.n_expert, L.n_slots,
                                               grp_ptr, grp_start, n_groups, ent_dst, ent_tok, Kp);
            K::quantize_q8_1_rows(x_norm, T, N, xq, st);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, hp.n_expert), n_ent, xq, escr, eparts, st);
        }
        if (prof_detail) {
            CK(cudaStreamSynchronize(st));
            const double mg_dt = ms_since(tp_mg0);
            if (!need_cpu) p_d_moe_gpu_core += mg_dt;
            else if (is_q5) p_d_moe_gpu_q5 += mg_dt;
            else p_d_moe_gpu_nv += mg_dt;
        }
        if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_gpu_ms += ms_since(tp_moe_r0); }

#if defined(STRATA_INKLING2_CPU_POOL)
        if (need_cpu) {
            const int T_all = T;
            const int cpu_step = cpu_rows > 0 ? cpu_rows : T_all;
            for (int sb0 = 0; sb0 < T_all; sb0 += cpu_step) {
            const int T = std::min(cpu_step, T_all - sb0);
            const int n_ent = T * Kp;
            int32_t* const h_ids = this->h_ids + (size_t) sb0 * Kp;
            auto* const eparts = this->eparts + (size_t) sb0 * Kp * N;
            if (sb0 > 0) {
                CK(cudaMemcpyAsync(h_h, x_norm + (size_t) sb0 * N, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToHost, st));
                CK(cudaEventRecord(ev_route, st));
            }
            namespace C = strata::kernels::cpu;
            const auto tp_wait0 = Clock::now();
            while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
            }
            if (prof) p_moe_wait_ms += ms_since(tp_wait0);
            const auto t0 = Clock::now();
            jobs.clear();
            int n_miss = 0;
            if (T == 1) {
                if (L.cpu_fmt.gu_act != 0) {
                    C::native_quant_act(L.cpu_fmt, h_h, acts);
                }
                const void* act0 = (L.cpu_fmt.gu_act == 0) ? (const void*) h_h : (const void*) acts;
                for (int e = 0; e < Kp; ++e) {
                    const int x = h_ids[e];
                    L.score[(size_t) x] += 1.0;
                    h_miss[e] = L.slot_of[(size_t) x] < 0 ? 1 : 0;
                    if (!h_miss[e]) continue;
                    ++n_miss;
                    C::ExpertJobMulti j;
                    j.blob = L.hblobs + (size_t) x * L.stride;
                    j.nact[0] = act0;
                    j.out[0] = cpu_out + (size_t) e * N;
                    j.nt = 1;
                    jobs.push_back(j);
                }
            } else {
                std::vector<char> quantized((size_t) T, 0);
                int open_job[128];
                std::fill_n(open_job, (int) hp.n_expert, -1);
                for (int e = 0; e < n_ent; ++e) {
                    const int x = h_ids[e];
                    L.score[(size_t) x] += 1.0;
                    h_miss[e] = L.slot_of[(size_t) x] < 0 ? 1 : 0;
                    if (!h_miss[e]) continue;
                    ++n_miss;
                    const int t = e / Kp;
                    if (L.cpu_fmt.gu_act != 0 && !quantized[(size_t) t]) {
                        C::native_quant_act(L.cpu_fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                        quantized[(size_t) t] = 1;
                    }
                    int jidx = open_job[x];
                    if (jidx < 0 || jobs[(size_t) jidx].nt == C::MAXT) {
                        C::ExpertJobMulti j;
                        j.blob = L.hblobs + (size_t) x * L.stride;
                        jidx = (int) jobs.size();
                        jobs.push_back(j);
                        open_job[x] = jidx;
                    }
                    C::ExpertJobMulti& j = jobs[(size_t) jidx];
                    j.nact[j.nt] = (L.cpu_fmt.gu_act == 0)
                                       ? (const void*) (h_h + (size_t) t * N)
                                       : (const void*) (acts + (size_t) t * C::kNativeActBytes);
                    j.out[j.nt] = cpu_out + (size_t) e * N;
                    ++j.nt;
                }
            }
            hy_cpu += n_miss;
            hy_gpu += n_ent - n_miss;
            if (prof_detail) {
                if (is_q5) { p_miss_q5 += n_miss; p_hit_q5 += (n_ent - n_miss); }
                else { p_miss_nv += n_miss; p_hit_nv += (n_ent - n_miss); }
            }
            if (n_miss > 0) {
                const auto tp_c0 = Clock::now();
                pool->run_split_multi_native(L.cpu_fmt, jobs.data(), (int) jobs.size());
                const double c_ms = ms_since(tp_c0);
                hy_cpu_ms += c_ms;
                if (prof) p_moe_cpu_ms += c_ms;
                if (prof_detail) {
                    if (is_q5) p_d_cpu_q5 += c_ms;
                    else p_d_cpu_nv += c_ms;
                }
                const auto tp_u0 = Clock::now();
                if (has_gpu) {
                    CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * sizeof(float), cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * sizeof(int32_t), cudaMemcpyHostToDevice, st));
                    k_merge_cpu<<<(unsigned) n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
                } else {
                    CK(cudaMemcpyAsync(eparts, cpu_out, (size_t) n_ent * N * sizeof(float), cudaMemcpyHostToDevice, st));
                }
                if (prof) { CK(cudaStreamSynchronize(st)); p_moe_up_ms += ms_since(tp_u0); }
            }
        if (sb0 + cpu_step < T_all) CK(cudaStreamSynchronize(st));
            }
        } else {
            hy_gpu += n_ent;
        }
#else
        if (need_cpu) {
            std::fprintf(stderr, "strata-inkling2: layer needs CPU pool but STRATA_NATIVE_EXPERTS is OFF\n");
            std::exit(1);
        }
        hy_gpu += n_ent;
#endif
        const auto tp_p0 = Clock::now();
        k_combine_experts<<<dim3((unsigned) ((N + 255) / 256), (unsigned) T), 256, 0, st>>>(eparts, ewts, moe_raw, N, Kp);
        if (prof && need_cpu) { CK(cudaStreamSynchronize(st)); p_moe_post_ms += ms_since(tp_p0); }
    }

    void rebalance(int max_swaps) {
        DeviceScope owner(device);
        if (role == Role::Outer && split_peer) split_peer->rebalance(max_swaps);
        if (replica || !hybrid) return;
        const int64_t ne = m->hp.n_expert;
        struct Cand {
            double gain;
            size_t li;
            int slot;
            int hot_x;
            int cold_y;
        };
        std::vector<Cand> cands;
        cands.reserve(m->layers.size() * 2);
        std::vector<int> order((size_t) ne);
        std::vector<int> victims;
        for (size_t li = 0; li < m->layers.size(); ++li) {
            if (!owns_phys((int) li)) continue;
            PhysLayer& L = m->layers[li];
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            const double weight = (L.L.gu_type == 13) ? 1.43 : 1.0;
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::partial_sort(order.begin(), order.begin() + std::min<int>((int) ne, S + 2), order.end(),
                              [&](int a, int b) { return L.score[(size_t) a] > L.score[(size_t) b]; });
            victims.resize((size_t) S);
            for (int s = 0; s < S; ++s) victims[(size_t) s] = s;
            std::partial_sort(victims.begin(), victims.begin() + std::min<int>(S, 2), victims.end(),
                              [&](int a, int b) {
                                  return L.score[(size_t) L.exp_in_slot[(size_t) a]] < L.score[(size_t) L.exp_in_slot[(size_t) b]];
                              });
            int hot_cand = -1;
            for (int i = 0; i < S; ++i) {
                const int x = order[(size_t) i];
                if (L.slot_of[(size_t) x] < 0) { hot_cand = x; break; }
            }
            if (hot_cand >= 0) {
                const int s = victims[0];
                const int y = L.exp_in_slot[(size_t) s];
                const double diff = L.score[(size_t) hot_cand] - L.score[(size_t) y];
                if (diff > 0.0) {
                    cands.push_back(Cand{diff * weight, li, s, hot_cand, y});
                }
            }
            for (int x = 0; x < ne; ++x) L.score[(size_t) x] *= 0.8;
        }
        if (cands.empty()) return;
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.gain > b.gain; });
        const int take = std::min<int>(max_swaps, (int) cands.size());
        for (int i = 0; i < take; ++i) {
            const Cand& c = cands[(size_t) i];
            PhysLayer& L = m->layers[c.li];
            L.slot_of[(size_t) c.cold_y] = -1;
            L.slot_of[(size_t) c.hot_x] = c.slot;
            L.exp_in_slot[(size_t) c.slot] = c.hot_x;
            CK(cudaMemcpyAsync(L.d_blobs + (size_t) c.slot * L.stride, L.hblobs + (size_t) c.hot_x * L.stride,
                               L.L.bytes, cudaMemcpyHostToDevice, cst));
            CK(cudaMemcpyAsync(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, cst));
        }
        CK(cudaEventRecord(ev_rebal, cst));
        has_pending_rebal = true;
    }

    // The metadata is local to one runtime. Only logical residual-buffer IDs cross the split.
    struct ForwardState {
        int T = 0;
        int next_blk_buf = 0;
        bool pp_prepared = false;
        bool capture_dspark = false, save_conv_trace = false;
        std::vector<float*> norm_blocks, prelude_norm_blocks, norm_bank;
    };

    void begin_forward(const int32_t* tokens, int T, ForwardState& state) {
        DeviceScope owner(device);
        state = ForwardState{};
        state.T = T;
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        // 1. Token embedding from host mmap + tok_embd_norm
        if (prof) ++p_calls;
        const auto tp_e0 = Clock::now();
        for (int t = 0; t < T; ++t) {
            const uint8_t* row = m->host_tok_embd + (size_t) tokens[t] * m->tok_embd_row_bytes;
            float* dst = h_tok_buf + (size_t) t * N;
            if (m->tok_embd_type == 30) {
                const uint16_t* bf = (const uint16_t*) row;
                for (int i = 0; i < N; ++i) {
                    uint32_t u = (uint32_t) bf[i] << 16;
                    std::memcpy(&dst[i], &u, 4);
                }
            } else if (m->tok_embd_type == 0) {
                std::memcpy(dst, row, (size_t) N * sizeof(float));
            } else {
                strata::quant::dequant_row((strata::quant::Type) m->tok_embd_type, row, dst, N);
            }
        }
        CK(cudaMemcpyAsync(ffn_inp, h_tok_buf, (size_t) T * N * sizeof(float), cudaMemcpyHostToDevice, st));
        pos0_host = (int32_t) pos;
        CK(cudaMemcpyAsync(pos0_d, &pos0_host, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(ffn_inp, m->tok_embd_norm, cur, N, hp.rms_eps);
        if (prof) { CK(cudaStreamSynchronize(st)); p_embd_ms += ms_since(tp_e0); }

    }

    void run_layers(ForwardState& state, int first_layer, int end_layer) {
        DeviceScope owner(device);
        const int T = state.T;
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const int64_t kv_dim = hp.kv_dim, q_dim = hp.q_dim;
        const int64_t n_core = hp.num_loops > 1 ? hp.core_end - hp.core_start : 0;
        const bool save_conv_trace = state.save_conv_trace;
        auto& norm_blocks = state.norm_blocks;
        int& next_blk_buf = state.next_blk_buf;
        auto push_block = [&](const float* src) {
            const int idx = next_blk_buf++;
            float* ndst = norm_blk_buf[idx];
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(src, nullptr, ndst, N, hp.rms_eps);
            norm_blocks.push_back(ndst);
        };

        auto& prelude_norm_blocks = state.prelude_norm_blocks;
        float* prelude_cur = prelude_cur_buf;
        auto& norm_bank = state.norm_bank;
        float* bank_last_norm = norm_blk_buf[8];     // cached normalized bank_last

        auto run_layer_pre = [&](int il) {
            const int ip = hp.loop_layer(il);
            PhysLayer& L = m->layers[(size_t) ip];
            const bool is_swa = hp.is_swa(il);

            const auto tp_d0 = Clock::now();
            bool has_msg = false;
            if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                const int pass = (il - (int) hp.core_start) / (int) n_core;
                if (ip == hp.core_start) {
                    if (pass == 0) {
                        prelude_norm_blocks = norm_blocks;
                        CK(cudaMemcpyAsync(prelude_cur, cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    } else {
                        norm_bank.clear();
                        for (size_t bi = prelude_norm_blocks.size() + 1; bi < norm_blocks.size(); ++bi) norm_bank.push_back(norm_blocks[bi]);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, nullptr, bank_last_norm, N, hp.rms_eps);
                        norm_blocks = prelude_norm_blocks;
                        CK(cudaMemcpyAsync(cur, prelude_cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    }
                }
                if (pass > 0) {
                    has_msg = true;
                    const int64_t row = 2 * (ip - hp.core_start);
                    const float* q_attn = m->loop_res_query + row * N;
                    const float* q_kv = m->loop_kv_res_query + row * N;
                    NormMixPtrs np{};
                    const int B = (int) norm_bank.size();
                    for (int i = 0; i < B; ++i) np.p[i] = norm_bank[(size_t) (B - 1 - i)];
                    np.p[B] = bank_last_norm;
                    MixOuts4 mo4{};
                    mo4.p[0] = msg0; mo4.p[1] = msg3; mo4.p[2] = msg1; mo4.p[3] = msg2;
                    k_mix_normed_4<<<dim3(4, (unsigned) T), kThreads, 0, st>>>(
                        np, B + 1, q_attn, q_kv, mo4, N, 1.0f / std::sqrt((float) N));
                }
            }

            const float* ink_src = in_k;
            const float* inv_src = in_v;
            if (state.capture_dspark && dspark) {
                for (int ti = 0; ti < 3; ++ti) if (il == dspark->dm->hp.target_layers[(size_t) ti])
                    CK(cudaMemcpyAsync(d_tgt_feat[ti], cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
            }
            if (L.attn_res_query) {
                float* outs[3] = {in_q, in_k, in_v};
                run_mix_normed(norm_blocks, cur, false, L.attn_res_query, outs, is_swa ? 1 : 3, T, is_swa ? 1 : 0);
                if (has_msg) {
                    run_controls_qkv(in_q, in_k, in_v, L, T);
                }
                if (is_swa && has_msg) {
                    k_rms_norm_3<<<dim3(3, (unsigned) T), kThreads, 0, st>>>(in_q, in_k, in_v, L.attn_norm, N, hp.rms_eps);
                } else if (is_swa) {
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_norm, in_q, N, hp.rms_eps);
                    ink_src = in_q;
                    inv_src = in_q;
                } else {
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                }
            } else {
                if (has_msg) {
                    run_controls_qkv(cur, cur, cur, L, T);
                    if (is_swa) {
                        k_rms_norm_3<<<dim3(3, (unsigned) T), kThreads, 0, st>>>(in_q, in_k, in_v, L.attn_norm, N, hp.rms_eps);
                    } else {
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_q, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_k, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_v, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                    }
                } else if (is_swa) {
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_norm, in_q, N, hp.rms_eps);
                    ink_src = in_q;
                    inv_src = in_q;
                } else {
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_q_inp_norm, in_q, N, hp.rms_eps);
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_k_inp_norm, in_k, N, hp.rms_eps);
                    k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(cur, L.attn_v_inp_norm, in_v, N, hp.rms_eps);
                }
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_mix += ms_since(tp_d0); }

            // Attention projections (quantize unique input vectors ONCE per layer)
            const auto tp_d1 = Clock::now();
            prep_q8(in_q, xq, T, N);
            const void* xq_k_src = (ink_src == in_q) ? xq : xq_k;
            const void* xq_v_src = (inv_src == in_q) ? xq : (inv_src == ink_src ? xq_k_src : xq_v);
            if (ink_src != in_q) prep_q8(ink_src, xq_k, T, N);
            if (inv_src != in_q && inv_src != ink_src) prep_q8(inv_src, xq_v, T, N);
            project_preq(in_q, xq, L.attn_sink_proj, sink, T, hp.n_head, N);
            float* pQ = Q;
            float* pGate = att_gate;
            float* pR = R;
            float* pK = K;
            float* pV = V;
            float* pDynK = dyn_k;
            float* pDynV = dyn_v;
            const int64_t rel_dim = hp.n_rel * hp.n_head;
            const int64_t dyn_dim = hp.n_conv * hp.n_head_kv;
            if (T == 1 && !has_msg && is_swa && L.attn_qgrkv.d != nullptr) {
                pQ = qgr_t1;
                pGate = qgr_t1 + q_dim;
                pR = qgr_t1 + 2 * q_dim;
                pK = pR + rel_dim;
                pV = pK + kv_dim;
                project_preq(in_q, xq, L.attn_qgrkv, qgr_t1, 1, L.attn_qgrkv.n_out, N);
            } else {
                if (T == 1 && L.attn_qgr.d != nullptr) {
                    pQ = qgr_t1;
                    pGate = qgr_t1 + q_dim;
                    pR = qgr_t1 + 2 * q_dim;
                    project_preq(in_q, xq, L.attn_qgr, qgr_t1, 1, L.attn_qgr.n_out, N);
                } else {
                    project_preq(in_q, xq, L.attn_q, Q, T, q_dim, N);
                    project_preq(in_q, xq, L.attn_rel, R, T, rel_dim, N);
                    project_preq(in_q, xq, L.attn_gate, att_gate, T, q_dim, N);
                }
                if (!is_swa && T == 1 && L.attn_kkc.d != nullptr) {
                    float* kbuf = qgr_t1 + 2 * q_dim + rel_dim;
                    pK = kbuf;
                    pDynK = kbuf + kv_dim;
                    project_preq(ink_src, xq_k_src, L.attn_kkc, kbuf, 1, L.attn_kkc.n_out, N);
                } else {
                    project_preq(ink_src, xq_k_src, L.attn_k, K, T, kv_dim, N);
                    if (!is_swa) project_preq(ink_src, xq_k_src, L.attn_k_conv_proj, dyn_k, T, dyn_dim, N);
                }
                if (!is_swa && T == 1 && L.attn_vvc.d != nullptr) {
                    float* vbuf = qgr_t1 + 2 * q_dim + rel_dim + kv_dim + dyn_dim;
                    pV = vbuf;
                    pDynV = vbuf + kv_dim;
                    project_preq(inv_src, xq_v_src, L.attn_vvc, vbuf, 1, L.attn_vvc.n_out, N);
                } else {
                    project_preq(inv_src, xq_v_src, L.attn_v, V, T, kv_dim, N);
                    if (!is_swa) project_preq(inv_src, xq_v_src, L.attn_v_conv_proj, dyn_v, T, dyn_dim, N);
                }
            }

            float* hst = conv_hist[(size_t) il];
            float* h_tr = save_conv_trace ? d_conv_trace + (size_t) trace_slot[(size_t) il] * layer_conv_floats : nullptr;
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_qkv += ms_since(tp_d1); }
            const auto tp_d2 = Clock::now();
            if (save_conv_trace)
                k_shortconv_kv_trace<<<(unsigned) ((2 * kv_dim + 255) / 256), 256, 0, st>>>(
                    pK, pV, hst, (int) kv_dim, T, h_tr, local_conv_floats);
            k_shortconv_kv<<<(unsigned) ((2 * kv_dim + 255) / 256), 256, 0, st>>>(
                pK, pV, L.attn_k_conv, L.attn_v_conv, is_swa ? nullptr : pDynK, is_swa ? nullptr : pDynV,
                hst, K_conv, V_conv, (int) kv_dim, (int) hp.n_head_kv, T);

            const int apply_tau = is_swa ? 0 : 1;
            k_clip_and_tau<<<dim3((unsigned) hp.n_head, (unsigned) T), 32, 0, st>>>(
                pQ, L.attn_q_clip, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
            k_clip_and_tau<<<dim3((unsigned) hp.n_head, (unsigned) T), 32, 0, st>>>(
                pR, L.attn_rel_clip, hp.rel_clip, (int) hp.n_rel, (int) hp.n_head, apply_tau, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);

            if (kv_q8) {
                k_clip_norm_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                    K_conv, V_conv, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                    (int) hp.n_head_kv, hp.qk_clip, hp.rms_eps, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0);
            } else {
                k_clip_and_tau<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(
                    K_conv, nullptr, hp.qk_clip, (int) hp.head_dim, (int) hp.n_head_kv, 0, pos0_d, (int) hp.log_scale_floor, hp.log_scale_alpha);
                k_v_rms_norm<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 32, 0, st>>>(V_conv, (int) hp.head_dim, (int) hp.n_head_kv, hp.rms_eps);
                k_kv_store<<<dim3((unsigned) ((kv_dim + 255) / 256), (unsigned) T), 256, 0, st>>>(
                    K_conv, V_conv, kc[(size_t) il], vc[(size_t) il], (int) kv_dim, pos0_d, kv_cap[(size_t) il], is_swa ? 1 : 0);
            }

            const int max_rel = is_swa ? std::min<int>((int) hp.n_swa, (int) hp.n_rel_ext) : (int) hp.n_rel_ext;
            k_rel_table<<<dim3((unsigned) hp.n_head, (unsigned) T), kThreads, 0, st>>>(
                pR, L.attn_rel_bias, sink, rel_tab, (int) hp.n_head, (int) hp.n_rel, (int) hp.n_rel_ext, max_rel, pos0_d, T);
            const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
            if (kv_q8) {
                if (hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128) {
                    if (T == 1) {
                        k_attn_q8_gqa3_split<<<dim3((unsigned) hp.n_head_kv, (unsigned) kAttnSplit), 128, 0, st>>>(
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, attn_split,
                            (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale, kAttnSplit);
                        k_attn_split_combine<<<(unsigned) hp.n_head, 128, 0, st>>>(attn_split, pGate, att, (int) hp.n_head, kAttnSplit);
                    } else {
                        k_attn_q8_gqa3_pp<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 128, 0, st>>>(
                            pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att,
                            (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
                    }
                } else {
                    k_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                        pQ, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], rel_tab, sink, pGate, att,
                        (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
                }
            } else {
                k_attn<<<dim3((unsigned) hp.n_head, (unsigned) T), 128, 0, st>>>(
                    pQ, kc[(size_t) il], vc[(size_t) il], rel_tab, sink, att,
                    (int) hp.n_head, (int) hp.n_head_kv, (int) hp.n_rel_ext, kv_cap[(size_t) il], is_swa ? 1 : 0, (int) hp.n_swa, pos0_d, kq_scale);
            }
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_attn += ms_since(tp_d2); }

            const auto tp_d3 = Clock::now();
            if (!kv_q8) {
                k_mul_sigmoid<<<(unsigned) (((int64_t) T * q_dim + 255) / 256), 256, 0, st>>>(att, pGate, (int64_t) T * q_dim);
            }
            project(att, L.attn_output, wo, T, N, q_dim);
            const bool do_push = (ip % hp.attn_res_block_size == 0);
            if (save_conv_trace)
                k_shortconv_trace<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                    wo, hst, N, 2 * (int) kv_dim, T, h_tr, local_conv_floats);
            k_shortconv<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(wo, L.attn_post_conv, nullptr, hst, ffn_inp, N, 2 * (int) kv_dim, 0, T, do_push ? nullptr : cur);
            if (do_push) push_block(cur);

            // MoE
            float* ffn_mix_out[1] = {in_ffn};
            run_mix_normed(norm_blocks, ffn_inp, false, L.ffn_res_query, ffn_mix_out, 1, T, 0);
            if (has_msg) run_control(in_ffn, msg3, L, 3, in_ffn, T);
            k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, L.ffn_norm, in_ffn, N, hp.rms_eps);
            if (prof_detail) { CK(cudaStreamSynchronize(st)); p_d_out += ms_since(tp_d3); }
        };

        auto advance_pre_meta = [&](int il) {
            const int ip = hp.loop_layer(il);
            if (n_core > 0 && ip >= hp.core_start && ip < hp.core_end) {
                const int pass = (il - (int) hp.core_start) / (int) n_core;
                if (ip == hp.core_start) {
                    if (pass == 0) {
                        prelude_norm_blocks = norm_blocks;
                    } else {
                        norm_bank.clear();
                        for (size_t bi = prelude_norm_blocks.size() + 1; bi < norm_blocks.size(); ++bi) norm_bank.push_back(norm_blocks[bi]);
                        norm_blocks = prelude_norm_blocks;
                    }
                }
            }
            if (ip % hp.attn_res_block_size == 0) {
                norm_blocks.push_back(norm_blk_buf[next_blk_buf++]);
            }
        };

        auto run_layer_post = [&](int il) {
            const int ip = hp.loop_layer(il);
            PhysLayer& L = m->layers[(size_t) ip];
            float* hst = conv_hist[(size_t) il];
            float* h_tr = save_conv_trace ? d_conv_trace + (size_t) trace_slot[(size_t) il] * layer_conv_floats : nullptr;
            const bool is_hy = (L.n_slots < hp.n_expert);
            run_moe(L, il, in_ffn, T);
            const auto tp_mp0 = Clock::now();
            if (save_conv_trace)
                k_shortconv_trace<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(
                    moe_raw, hst, N, 2 * (int) kv_dim + N, T, h_tr, local_conv_floats);
            k_shortconv<<<(unsigned) ((N + 255) / 256), 256, 0, st>>>(moe_raw, L.ffn_post_conv, nullptr, hst, cur, N, 2 * (int) kv_dim + N, 0, T, ffn_inp);
            if (prof) {
                CK(cudaStreamSynchronize(st));
                if (is_hy) p_moe_post_ms += ms_since(tp_mp0);
                else if (ip >= hp.core_start && ip < hp.core_end) p_core_ms += ms_since(tp_mp0);
            }
        };

        const int core_end_virt = (int) (hp.core_start + hp.num_loops * n_core);
        if (!state.pp_prepared) {
            pp_streaming = (hybrid && T >= pp_stream_t && pp_stage[0] != nullptr);
            if (pp_streaming) pp_begin(core_end_virt, core_all_gpu ? 1 : 0);
            state.pp_prepared = true;
        }
        // Both ranges use the unchanged layer kernels. Copies, position/token updates,
        // the cross-device handoff and the output head remain outside these graphs.
        const int outer_range = first_layer == 0 && end_layer == hp.core_start ? 0 :
                                first_layer == core_end_virt && end_layer == hp.n_layer ? 1 : -1;
        if (split_outer_graph && T == 1 && outer_range >= 0 && !state.capture_dspark &&
            !save_conv_trace && outer_graph_eligible()) {
            const std::vector<float*> expected_in = outer_range == 0 ? std::vector<float*>{} :
                                                    std::vector<float*>{norm_blk_buf[0], norm_blk_buf[5], norm_blk_buf[6]};
            const std::vector<float*> expected_out = outer_range == 0 ? std::vector<float*>{norm_blk_buf[0]} :
                std::vector<float*>{norm_blk_buf[0], norm_blk_buf[5], norm_blk_buf[6], norm_blk_buf[7]};
            const int next_in = outer_range == 0 ? 0 : 7, next_out = outer_range == 0 ? 1 : 8;
            if (norm_blocks == expected_in && next_blk_buf == next_in) {
                auto& graph = outer_graph[outer_range];
                if (!graph && !outer_graph_failed[outer_range]) {
                    const ForwardState saved_state = state;
                    const int64_t saved_hits = hy_gpu, saved_misses = hy_cpu;
                    const double saved_cpu_ms = hy_cpu_ms;
                    cudaGraph_t captured = nullptr;
                    cudaGraphExec_t candidate = nullptr;
                    cudaError_t status = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                    if (status == cudaSuccess) {
                        for (int il = first_layer; il < end_layer; ++il) {
                            run_layer_pre(il);
                            run_layer_post(il);
                        }
                        status = cudaStreamEndCapture(st, &captured);
                    }
                    if (status == cudaSuccess && (!captured || norm_blocks != expected_out || next_blk_buf != next_out))
                        status = cudaErrorInvalidValue;
                    if (status == cudaSuccess && captured)
                        status = cudaGraphInstantiate(&candidate, captured, nullptr, nullptr, 0);
                    if (captured) cudaGraphDestroy(captured);
                    // Capture only records device work, but its host calls advance these
                    // fields. Undo them before either the first launch or eager fallback.
                    state = saved_state;
                    hy_gpu = saved_hits; hy_cpu = saved_misses; hy_cpu_ms = saved_cpu_ms;
                    if (status == cudaSuccess && candidate) {
                        graph = candidate;
                    } else {
                        if (candidate) cudaGraphExecDestroy(candidate);
                        outer_graph_failed[outer_range] = true;
                        (void) cudaGetLastError();
                        std::fprintf(stderr, "strata-inkling2: outer %s graph capture/instantiate failed on device %d: %s; using eager range\n",
                                     outer_range == 0 ? "prefix" : "suffix", device, cudaGetErrorString(status));
                    }
                }
                if (graph) {
                    // A launch error is surfaced: after execution may have started, replaying
                    // eager could apply the recurrent updates twice and is not a safe fallback.
                    const cudaError_t status = cudaGraphLaunch(graph, st);
                    if (status != cudaSuccess)
                        std::fprintf(stderr, "strata-inkling2: outer %s graph launch failed on device %d: %s\n",
                                     outer_range == 0 ? "prefix" : "suffix", device, cudaGetErrorString(status));
                    CK(status);
                    norm_blocks = expected_out;
                    next_blk_buf = next_out;
                    // Six all-resident layers x top-8 = 48 lookups, counted once on
                    // the first successful launch as well as on every replay.
                    hy_gpu += (int64_t) (end_layer - first_layer) * hp.n_expert_used;
                    return;
                }
            }
        }
        for (int il = first_layer; il < end_layer; ++il) {
            if (!owns_virtual(il)) {
                std::fprintf(stderr, "strata-inkling2: virtual layer %d belongs to the other split runtime\n", il);
                std::exit(1);
            }
            const bool core_graph_allowed = role == Role::All || (role == Role::Core && split_core_graph);
            const bool graph_shape = T == 1 || (T == verify_max_t && state.capture_dspark && save_conv_trace);
            if (core_graph_allowed && graph_shape && core_all_gpu && !prof_detail && il == hp.core_start && n_core > 0) {
                cudaGraphExec_t& cg = T != 1 ? verify_core_graph : (state.capture_dspark ? capture_core_graph : core_graph);
                if (!cg) {
                    cudaGraph_t g = nullptr;
                    auto graph_check = [&](cudaError_t status, const char* phase) {
                        if (status != cudaSuccess && role == Role::Core)
                            std::fprintf(stderr, "strata-inkling2: split core graph %s failed on device %d: %s\n",
                                         phase, device, cudaGetErrorString(status));
                        CK(status);
                    };
                    graph_check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
                    for (int cl = il; cl < core_end_virt; ++cl) {
                        run_layer_pre(cl);
                        run_layer_post(cl);
                    }
                    graph_check(cudaStreamEndCapture(st, &g), "end capture");
                    graph_check(cudaGraphInstantiate(&cg, g, nullptr, nullptr, 0), "instantiate");
                    CK(cudaGraphDestroy(g));
                } else {
                    norm_blocks = {norm_blk_buf[0], norm_blk_buf[5], norm_blk_buf[6]};
                    next_blk_buf = 7;
                }
                const auto tp_cg0 = Clock::now();
                CK(cudaGraphLaunch(cg, st));
                if (prof) { CK(cudaStreamSynchronize(st)); p_core_ms += ms_since(tp_cg0); }
                il = core_end_virt - 1;
                continue;
            }
            const bool is_outer = (il < hp.core_start || il >= core_end_virt);
            const bool use_pre_graph = (role == Role::All && T == 1 && hybrid && !prof_detail && !pp_streaming && !state.capture_dspark);
            const auto tp_ly0 = Clock::now();
            if (use_pre_graph && (layer_pre_graph[(size_t) il] || !graph_oom)) {
                if (!layer_pre_graph[(size_t) il]) {
                    const auto saved_nb = norm_blocks;
                    const int saved_next = next_blk_buf;
                    const auto saved_pnb = prelude_norm_blocks;
                    const auto saved_bank = norm_bank;
                    cudaGraph_t g = nullptr;
                    cudaError_t ce = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                    if (ce == cudaSuccess) {
                        run_layer_pre(il);
                        ce = cudaStreamEndCapture(st, &g);
                    }
                    cudaGraphExec_t ge = nullptr;
                    if (ce == cudaSuccess && g) ce = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
                    if (g) cudaGraphDestroy(g);
                    if (ce == cudaSuccess && ge) {
                        layer_pre_graph[(size_t) il] = ge;
                        CK(cudaGraphLaunch(layer_pre_graph[(size_t) il], st));
                    } else {
                        (void) cudaGetLastError();
                        graph_oom = true;
                        norm_blocks = saved_nb;
                        next_blk_buf = saved_next;
                        prelude_norm_blocks = saved_pnb;
                        norm_bank = saved_bank;
                        run_layer_pre(il);
                    }
                } else {
                    advance_pre_meta(il);
                    CK(cudaGraphLaunch(layer_pre_graph[(size_t) il], st));
                }
            } else {
                run_layer_pre(il);
            }
            if (prof) {
                CK(cudaStreamSynchronize(st));
                if (is_outer) p_attn_ms += ms_since(tp_ly0);
                else p_core_ms += ms_since(tp_ly0);
            }
            run_layer_post(il);
        }

    }

    int32_t finish_head(ForwardState& state, bool copy_logits, std::vector<float>* out_logits,
                        bool all_row_logits = false, std::vector<int32_t>* out_picks = nullptr) {
        DeviceScope owner(device);
        const int T = state.T;
        const auto& hp = m->hp;
        const int N = (int) hp.n_embd;
        const auto& norm_blocks = state.norm_blocks;
        // Output head
        const auto tp_h0 = Clock::now();
        float* out_head[1] = {in_ffn};
        run_mix_normed(norm_blocks, cur, false, m->output_res_query, out_head, 1, T, 0);
        k_rms_norm<<<(unsigned) T, kThreads, 0, st>>>(in_ffn, m->output_norm, cur, N, hp.rms_eps);
        if (state.capture_dspark && dspark)
            CK(cudaMemcpyAsync(d_tgt_feat[3], cur, (size_t) T * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
        const int rows = all_row_logits || out_picks ? T : 1;
        const float* head_h = rows == T ? cur : cur + (size_t) (T - 1) * N;
        project(head_h, m->output_weight, logits, rows, hp.vocab_unpadded, N);
        const int64_t total_logits = (int64_t) rows * hp.vocab_unpadded;
        k_softcap<<<(unsigned) ((total_logits + 255) / 256), 256, 0, st>>>(logits, total_logits, hp.softcap);
        k_argmax<<<(unsigned) rows, 256, 0, st>>>(logits, pick_d, (int) hp.vocab_unpadded);
        std::vector<int32_t> picks((size_t) rows);
        if (copy_logits && out_logits) {
            out_logits->resize((size_t) total_logits);
            CK(cudaMemcpyAsync(out_logits->data(), logits, (size_t) total_logits * sizeof(float), cudaMemcpyDeviceToHost, st));
        }
        CK(cudaMemcpyAsync(picks.data(), pick_d, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (out_picks) *out_picks = picks;
        const int32_t next_id = picks.back();
        if (prof) p_head_ms += ms_since(tp_h0);
        pos += T;
        return next_id;
    }

    // A synchronization at each boundary makes the portable host buffer reusable on either
    // device without peer access, cross-device event assumptions or foreign device pointers.
    void handoff(ForwardState& source, Engine& destination, ForwardState& target) {
        const int T = source.T;
        const size_t elems = (size_t) T * m->hp.n_embd;
        float* host = role == Role::Outer ? split_handoff : split_peer->split_handoff;
        std::vector<int> block_ids;
        for (float* p : source.norm_blocks) {
            int id = -1;
            for (int i = 0; i < 9; ++i) if (p == norm_blk_buf[i]) id = i;
            if (id < 0) { std::fprintf(stderr, "strata-inkling2: split residual has no logical ID\n"); std::exit(1); }
            block_ids.push_back(id);
        }
        if (!host || T <= 0 || T > tmax || T > destination.tmax || block_ids.size() > 3 || source.next_blk_buf > 8) {
            std::fprintf(stderr, "strata-inkling2: invalid split residual metadata\n"); std::exit(1);
        }
        const std::vector<int> expected_ids = role == Role::Outer ? std::vector<int>{0} : std::vector<int>{0, 5, 6};
        const int expected_next = role == Role::Outer ? 1 : 7;
        if (block_ids != expected_ids || source.next_blk_buf != expected_next) {
            std::fprintf(stderr, "strata-inkling2: split boundary residual IDs do not match the 6/36/6 layer schedule\n");
            std::exit(1);
        }
        {
            DeviceScope owner(device);
            CK(cudaMemcpyAsync(host, cur, elems * sizeof(float), cudaMemcpyDeviceToHost, st));
            for (size_t i = 0; i < block_ids.size(); ++i)
                CK(cudaMemcpyAsync(host + (i + 1) * elems, norm_blk_buf[block_ids[i]], elems * sizeof(float), cudaMemcpyDeviceToHost, st));
            if (role == Role::Core && source.capture_dspark && dspark) {
                float* features = destination.split_features;
                for (int i = 0; i < 2; ++i)
                    CK(cudaMemcpyAsync(features + (size_t) i * elems, d_tgt_feat[i + 1], elems * sizeof(float), cudaMemcpyDeviceToHost, st));
            }
            CK(cudaStreamSynchronize(st));
        }
        target.T = T;
        target.capture_dspark = source.capture_dspark;
        target.save_conv_trace = source.save_conv_trace;
        target.next_blk_buf = source.next_blk_buf;
        target.norm_blocks.clear();
        // Prelude/bank data only lives inside the core range; both boundaries are outside it.
        target.prelude_norm_blocks.clear();
        target.norm_bank.clear();
        destination.pos = pos;
        {
            DeviceScope owner(destination.device);
            CK(cudaMemcpyAsync(destination.cur, host, elems * sizeof(float), cudaMemcpyHostToDevice, destination.st));
            for (size_t i = 0; i < block_ids.size(); ++i) {
                float* p = destination.norm_blk_buf[block_ids[i]];
                CK(cudaMemcpyAsync(p, host + (i + 1) * elems, elems * sizeof(float), cudaMemcpyHostToDevice, destination.st));
                target.norm_blocks.push_back(p);
            }
            const int32_t pos32 = (int32_t) pos;
            CK(cudaMemcpyAsync(destination.pos0_d, &pos32, sizeof pos32, cudaMemcpyHostToDevice, destination.st));
            if (role == Role::Core && source.capture_dspark && dspark) {
                for (int i = 0; i < 2; ++i)
                    CK(cudaMemcpyAsync(destination.d_tgt_feat[i + 1], destination.split_features + (size_t) i * elems,
                                       elems * sizeof(float), cudaMemcpyHostToDevice, destination.st));
            }
            CK(cudaStreamSynchronize(destination.st));
        }
    }

    int32_t forward(const int32_t* tokens, int T, bool copy_logits = false, std::vector<float>* out_logits = nullptr,
                    bool all_row_logits = false, bool capture_dspark = false, int verify_k_save = 0,
                    std::vector<int32_t>* out_picks = nullptr) {
        if (T <= 0 || T > tmax || pos < 0 || pos + T > ctx || ((all_row_logits || out_picks) && T > verify_max_t) ||
            (verify_k_save > 0 && (!dspark || verify_k_save != T - 1 || T > verify_max_t))) {
            std::fprintf(stderr, "strata-inkling2: invalid forward/verify extent\n"); std::exit(1);
        }
        if (verify_k_save > 0) {
            backup_verify(verify_k_save);
            if (role == Role::Outer && split_peer) split_peer->backup_verify(verify_k_save);
        }
        if (role == Role::Outer && split_peer) {
            split_peer->prof = prof;
            split_peer->prof_detail = prof_detail;
            if (prof) ++split_peer->p_calls;
        }
        ForwardState state;
        begin_forward(tokens, T, state);
        state.capture_dspark = capture_dspark;
        state.save_conv_trace = verify_k_save > 0;
        if (role == Role::Outer && split_peer) {
            const int begin = (int) m->hp.core_start;
            const int end = (int) (m->hp.core_start + m->hp.num_loops * (m->hp.core_end - m->hp.core_start));
            run_layers(state, 0, begin);
            ForwardState core_state;
            handoff(state, *split_peer, core_state);
            split_peer->run_layers(core_state, begin, end);
            split_peer->handoff(core_state, *this, state);
            run_layers(state, end, (int) m->hp.n_layer);
        } else {
            run_layers(state, 0, (int) m->hp.n_layer);
        }
        const int32_t next = finish_head(state, copy_logits, out_logits, all_row_logits, out_picks);
        if (role == Role::Outer && split_peer) split_peer->pos = pos;
        return next;
    }

};

bool place_replica_experts(Model& m, Engine& E, int remote_device, size_t hot_budget, bool async, std::string& err) {
    DeviceScope owner(E.device);
    auto r = std::make_unique<ExpertReplica>();
    r->device = remote_device;
    r->async = async;
    const int ne = (int) m.hp.n_expert;
    const size_t entries = (size_t) E.tmax * m.hp.n_expert_used;
    const size_t qbytes = (size_t) E.tmax * (m.hp.n_embd / 32) * 36;
    const size_t pbytes = entries * m.hp.n_embd * sizeof(float);
    const size_t scratch_bytes = K::native_expert_scratch_bytes(entries, m.hp.n_ff);
    const size_t reserve = (size_t) 512 << 20;
    size_t strides = 0, full_bytes = 0;
    for (const auto& L : m.layers) {
        if (L.gu_type == 30 || !L.stride) { err = "replica requires native GPU expert layouts for every physical layer"; return false; }
        strides += L.stride;
        full_bytes += (size_t) ne * L.stride;
    }
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    const size_t local_extra = pbytes + entries * sizeof(int32_t) + m.layers.size() * ne * sizeof(int32_t) +
        (async ? (m.layers.size() * ne + 3) * sizeof(unsigned long long) : 0);
    if (free_bytes < reserve + local_extra) { err = "replica dense GPU lacks workspace reserve after full-capacity KV allocation"; return false; }
    hot_budget = std::min(hot_budget, free_bytes - reserve - local_extra);
    const int slots = (int) std::min<size_t>(ne, hot_budget / strides);
    r->hot_bytes = (size_t) slots * strides;
    r->full_bytes = full_bytes;
    const size_t remote_extra = qbytes + pbytes + scratch_bytes + entries * 3 * sizeof(int32_t) +
        (m.layers.size() * ne + ne + 2) * sizeof(int32_t) + ne * sizeof(unsigned long long);
    { DeviceScope remote(remote_device);
      CK(cudaMemGetInfo(&free_bytes, &total));
      if (free_bytes < full_bytes + remote_extra + reserve) {
          err = "replica GPU cannot hold all experts plus scratch and 512 MiB reserve"; return false;
      }
      CK(cudaStreamCreateWithFlags(&r->stream, cudaStreamNonBlocking));
      if (async) CK(cudaEventCreateWithFlags(&r->remote_done, cudaEventDisableTiming));
      r->alloc(&r->xq, qbytes); r->alloc(&r->scratch, scratch_bytes);
      r->alloc(&r->ids, entries * sizeof(int32_t));
      r->alloc(&r->starts, (ne+1) * sizeof(int32_t)); r->alloc(&r->groups, sizeof(int32_t));
      r->alloc(&r->dst, entries * sizeof(int32_t)); r->alloc(&r->tok, entries * sizeof(int32_t));
      r->alloc(&r->ptrs, ne * sizeof(unsigned long long)); r->alloc(&r->parts, pbytes);
      std::vector<uint8_t> tmp;
      std::vector<int32_t> identity((size_t) ne);
      for (int e = 0; e < ne; ++e) identity[e] = e;
      for (auto& L : m.layers) {
          uint8_t* blobs = nullptr; int32_t* map = nullptr;
          r->alloc(&blobs, (size_t) ne * L.stride); r->alloc(&map, ne * sizeof(int32_t));
          r->blobs.push_back(blobs); r->maps.push_back(map); r->cold_of.push_back(identity);
          tmp.resize(L.L.bytes);
          for (int e = 0; e < ne; ++e) {
              build_expert_blob(L, e, tmp.data());
              CK(cudaMemcpy(blobs + (size_t) e * L.stride, tmp.data(), L.L.bytes, cudaMemcpyHostToDevice));
          }
          CK(cudaMemcpy(map, identity.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice));
      }
    }
    CK(cudaHostAlloc(&r->host_ids, entries * sizeof(int32_t), cudaHostAllocPortable));
    CK(cudaHostAlloc(&r->host_miss, entries * sizeof(int32_t), cudaHostAllocPortable));
    CK(cudaHostAlloc(&r->host_xq, qbytes, cudaHostAllocPortable));
    CK(cudaHostAlloc(&r->host_parts, pbytes, cudaHostAllocPortable));
    if (async) {
        CK(cudaEventCreateWithFlags(&r->input_ready, cudaEventDisableTiming));
        CK(cudaMalloc(&r->frequencies, m.layers.size() * ne * sizeof(unsigned long long)));
        CK(cudaMalloc(&r->hits, 3 * sizeof(unsigned long long)));
        CK(cudaMemset(r->frequencies, 0, m.layers.size() * ne * sizeof(unsigned long long)));
        CK(cudaMemset(r->hits, 0, 3 * sizeof(unsigned long long)));
    }
    CK(cudaMalloc(&r->local_miss, entries * sizeof(int32_t)));
    CK(cudaMalloc(&r->local_parts, pbytes));
    for (auto& L : m.layers) {
        L.n_slots = slots;
        L.score.assign(ne, 0); L.slot_of.assign(ne, -1);
        CK(cudaMalloc(&L.d_slot_of, ne * sizeof(int32_t))); m.record_alloc(L.d_slot_of);
        CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (slots) { CK(cudaMalloc(&L.d_blobs, (size_t) slots * L.stride)); m.record_alloc(L.d_blobs); }
    }
    E.placed_slots = (int) m.layers.size() * ne;
    E.placed_bytes = full_bytes + r->hot_bytes;
    E.core_all_gpu = false; // no layer graphs: routing and pinned relay remain eager
    CK(cudaMemGetInfo(&E.vram_free_after_place, &total));
    std::fprintf(stderr, "REPLICA {\"event\":\"READY\",\"dense_device\":%d,\"replica_device\":%d,\"full_experts\":%d,\"hot_capacity\":%d,\"hot_loaded\":0,\"full_bytes\":%llu,\"hot_bytes\":%llu,\"reserve_bytes\":%llu,\"kv_capacity\":%d,\"cold_start\":true,\"async\":%s}\n",
        E.device, remote_device, E.placed_slots, slots * (int) m.layers.size(),
        (unsigned long long) full_bytes, (unsigned long long) r->hot_bytes, (unsigned long long) reserve, E.ctx, async ? "true" : "false");
    E.replica = std::move(r);
    return true;
}

bool place_experts(Model& m, Engine& E, double vram_gib, int threads, std::string& err) {
    DeviceScope owner(E.device);
    const auto t0 = Clock::now();
    const auto& hp = m.hp;
    const int ne = (int) hp.n_expert;
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) 256 << 20;
    size_t budget = vram_gib >= 0.0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                    : (fr > reserve ? fr - reserve : 0);

    size_t stage_mb = 96;
    if (const char* v = std::getenv("STRATA_PP_STAGE_MB")) stage_mb = (size_t) std::max(0, std::atoi(v));
    size_t max_exp_stride = 0;
    size_t all_exps_bytes = 0;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        if (!E.owns_phys(ip)) continue;
        if (m.layers[(size_t) ip].gu_type == 30) continue;
        max_exp_stride = std::max(max_exp_stride, m.layers[(size_t) ip].stride);
        all_exps_bytes += (size_t) ne * m.layers[(size_t) ip].stride;
    }
    // Keep extra ~144 MiB (400 MiB total) headroom when hybrid-offloading so 48 CUDA pre-graphs + desktop fit OOM-free
    if (vram_gib < 0.0 && budget < all_exps_bytes) {
        const size_t extra_resv = (size_t) 144 << 20;
        budget = budget > extra_resv ? (budget - extra_resv) : 0;
    }
    if (budget < all_exps_bytes && stage_mb > 0 && max_exp_stride > 0) {
        size_t req_stage = stage_mb << 20;
        if (req_stage < max_exp_stride) req_stage = max_exp_stride;
        if (budget > req_stage * 2 + ((size_t) 256 << 20)) {
            if (cudaMalloc(&E.pp_stage[0], req_stage) == cudaSuccess &&
                cudaMalloc(&E.pp_stage[1], req_stage) == cudaSuccess) {
                E.pp_stage_bytes = req_stage;
                budget -= req_stage * 2;
            } else {
                (void) cudaGetLastError();
                if (E.pp_stage[0]) { cudaFree(E.pp_stage[0]); E.pp_stage[0] = nullptr; }
                if (E.pp_stage[1]) { cudaFree(E.pp_stage[1]); E.pp_stage[1] = nullptr; }
            }
        }
    }
    size_t fr_before_cpu = 0;
    CK(cudaMemGetInfo(&fr_before_cpu, &tot));
    const int cpu_t = (E.pp_stage[0] && E.pp_stream_t > 1) ? std::min(E.tmax, E.pp_stream_t - 1) : E.tmax;
    E.init_cpu_bufs(cpu_t);
    {
        size_t fr_cpu = 0;
        CK(cudaMemGetInfo(&fr_cpu, &tot));
        const size_t used_cpu = (fr_before_cpu > fr_cpu) ? (fr_before_cpu - fr_cpu) : 0;
        if (budget > used_cpu) budget -= used_cpu;
        else budget = 0;
    }

    size_t used = 0;
    std::vector<int> slots(m.layers.size(), 0);

    // Priority 1: Looped core NVFP4 layers (blk.6..17, executed 3x/token!)
    std::vector<int> tier1, tier2;
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        if (!E.owns_phys(ip)) continue;
        if (m.layers[(size_t) ip].gu_type == 30) continue;   // BF16 stays CPU-resident
        if (ip >= hp.core_start && ip < hp.core_end) tier1.push_back(ip);
        else tier2.push_back(ip);
    }

    size_t tier1_full = 0;
    for (int ip : tier1) tier1_full += (size_t) ne * m.layers[(size_t) ip].stride;
    if (!tier1.empty()) {
        if (used + tier1_full <= budget) {
            for (int ip : tier1) slots[(size_t) ip] = ne;
            used += tier1_full;
        } else {
            // Priority 1 strictly maximizes tier1 (core runs 3x/tok & uses compact NVFP4);
            // optional STRATA_TIER2_FLOOR can reserve outer slots when desired.
            int t2_floor = 0;
            if (const char* v = std::getenv("STRATA_TIER2_FLOOR")) t2_floor = std::max(0, std::atoi(v));
            size_t sum_tier2_stride = 0;
            for (int ip : tier2) sum_tier2_stride += m.layers[(size_t) ip].stride;
            size_t tier2_resv = (size_t) t2_floor * sum_tier2_stride;
            const size_t per_layer_stride = m.layers[(size_t) tier1[0]].stride;
            const size_t min_tier1_bytes = (size_t) std::max(1, ne / 2) * tier1.size() * per_layer_stride;
            if (tier2.empty() || budget <= used + min_tier1_bytes) {
                tier2_resv = 0;
            } else if (budget - used - min_tier1_bytes < tier2_resv) {
                tier2_resv = budget - used - min_tier1_bytes;
            }
            const size_t t1_budget = (budget > used + tier2_resv) ? (budget - used - tier2_resv) : (budget - used);
            const int s = (int) std::min<size_t>((size_t) ne, t1_budget / (tier1.size() * per_layer_stride));
            for (int ip : tier1) slots[(size_t) ip] = s;
            used += (size_t) s * tier1.size() * per_layer_stride;
        }
    }

    // Priority 2: Distribute remaining VRAM equally across tier2, then greedily fill leftover budget
    if (!tier2.empty() && budget > used) {
        size_t sum_tier2_stride = 0;
        for (int ip : tier2) sum_tier2_stride += m.layers[(size_t) ip].stride;
        const int s = sum_tier2_stride > 0 ? (int) std::min<size_t>((size_t) ne, (budget - used) / sum_tier2_stride) : 0;
        if (s > 0) {
            for (int ip : tier2) slots[(size_t) ip] = s;
            used += (size_t) s * sum_tier2_stride;
        }
        bool added = true;
        while (added && used < budget) {
            added = false;
            for (int ip : tier2) {
                const size_t st_b = m.layers[(size_t) ip].stride;
                if (slots[(size_t) ip] < ne && used + st_b <= budget) {
                    slots[(size_t) ip] += 1;
                    used += st_b;
                    added = true;
                }
            }
        }
    }

    std::vector<uint8_t> tmp;
    bool need_cpu = false;
    int total_vram_experts = 0;
    size_t actual_used = 0;
    size_t pinned_used = 0;
    size_t max_pin_bytes = (size_t) 7168 << 20;   // 7.0 GiB cap covers 100% of tier1 (or all tier2 at 64K) without swap thrashing
    if (const char* v = std::getenv("STRATA_MAX_PIN_MB")) max_pin_bytes = (size_t) std::max(0, std::atoi(v)) << 20;
    std::vector<int> alloc_order;
    alloc_order.reserve((size_t) hp.n_phys);
    for (int ip : tier1) alloc_order.push_back(ip);
    for (int ip : tier2) alloc_order.push_back(ip);
    for (int ip = 0; ip < hp.n_phys; ++ip) {
        if (E.owns_phys(ip) && m.layers[(size_t) ip].gu_type == 30) alloc_order.push_back(ip);
    }
    for (int ip : alloc_order) {
        PhysLayer& L = m.layers[(size_t) ip];
        int s = slots[(size_t) ip];
        L.slot_of.assign((size_t) ne, -1);
        L.score.assign((size_t) ne, 0.0);

        if (s > 0) {
            void* dbl = nullptr;
            while (s > 0 && cudaMalloc(&dbl, (size_t) s * L.stride) != cudaSuccess) {
                (void) cudaGetLastError();
                --s;
            }
            if (s > 0 && dbl) {
                L.d_blobs = (uint8_t*) dbl;
                m.record_alloc(L.d_blobs);
                CK(cudaMalloc(&L.d_slot_of, (size_t) ne * sizeof(int32_t)));
                m.record_alloc(L.d_slot_of);
                actual_used += (size_t) s * L.stride;
            } else {
                s = 0;
            }
        }
        slots[(size_t) ip] = s;
        L.n_slots = s;
        L.exp_in_slot.assign((size_t) s, -1);

        if (s < ne) {
            need_cpu = true;
            const size_t hb_bytes = (size_t) ne * L.stride;
            const bool want_pin = (std::getenv("STRATA_NO_PIN_EXPERTS") == nullptr) && (pinned_used + hb_bytes <= max_pin_bytes);
            if (want_pin && cudaHostAlloc(&L.hblobs, hb_bytes, cudaHostAllocDefault) == cudaSuccess) {
                m.host_pinned_allocs.push_back(L.hblobs);
                pinned_used += hb_bytes;
            } else {
                (void) cudaGetLastError();
                L.hblobs = (uint8_t*) std::malloc(hb_bytes);
                if (!L.hblobs) {
                    err = "host malloc failed for layer " + std::to_string(ip);
                    return false;
                }
                m.host_allocs.push_back(L.hblobs);
            }
            for (int e = 0; e < ne; ++e) build_expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        }

        if (s > 0) {
            tmp.resize(L.L.bytes);
            for (int k = 0; k < s; ++k) {
                const uint8_t* src = L.hblobs ? (L.hblobs + (size_t) k * L.stride) : (build_expert_blob(L, k, tmp.data()), tmp.data());
                CK(cudaMemcpy(L.d_blobs + (size_t) k * L.stride, src, L.L.bytes, cudaMemcpyHostToDevice));
                L.slot_of[(size_t) k] = k;
                L.exp_in_slot[(size_t) k] = k;
            }
            CK(cudaMemcpy(L.d_slot_of, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice));
            total_vram_experts += s;
        }
        if (s > 0 && s < ne) E.hybrid = true;
    }
    used = actual_used;

#if defined(STRATA_INKLING2_CPU_POOL)
    if (need_cpu) {
        int nw = threads > 0 ? threads : 0;
        E.pool = std::make_unique<strata::kernels::cpu::ExpertPool>(nw, true, true);
        E.pool_workers = E.pool->workers();
        std::fprintf(stderr, "strata-inkling2: %s\n", E.pool->placement_line().c_str());
    }
#else
    if (need_cpu) {
        err = "model experts exceed VRAM and STRATA_NATIVE_EXPERTS is OFF";
        return false;
    }
#endif

    E.core_all_gpu = !tier1.empty() && slots[(size_t) tier1[0]] == ne;
    for (int ip : tier1) if (slots[(size_t) ip] != ne) E.core_all_gpu = false;

    size_t fr_after = 0, tot_after = 0;
    CK(cudaMemGetInfo(&fr_after, &tot_after));
    E.placed_slots = total_vram_experts;
    E.placed_bytes = used;
    E.vram_free_after_place = fr_after;

    int tier2_min = ne, tier2_max = 0;
    for (int ip : tier2) {
        tier2_min = std::min(tier2_min, slots[(size_t) ip]);
        tier2_max = std::max(tier2_max, slots[(size_t) ip]);
    }
    std::fprintf(stderr,
                 "strata-inkling2: placed %d/%lld experts in VRAM (%.2f GiB, tier1=%d/128/layer, tier2=%d..%d/128/layer, free=%.0f MiB) in %.2f s\n",
                 total_vram_experts, (long long) alloc_order.size() * ne, (double) used / (1 << 30),
                 tier1.empty() ? 0 : slots[(size_t) tier1[0]],
                 tier2.empty() ? 0 : tier2_min, tier2.empty() ? 0 : tier2_max,
                 (double) fr_after / (1 << 20),
                 ms_since(t0) / 1000.0);
    return true;
}

std::vector<int32_t> parse_id_list(const std::string& s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::stringstream ws(item);
        int32_t id;
        while (ws >> id) out.push_back(id);
    }
    return out;
}


/// Shared autoregressive / DSpark speculative decode loop for both `--serve` and `--bench`/CLI.
template <typename EmitFn, typename StopCheckFn>
void decode_tokens_loop(Engine& E, DsparkEngine* dspark, int draft_max, int32_t first_tok, int limit,
                        const SamplingParams& smp, std::mt19937_64& rng,
                        std::vector<int32_t>& hist, std::vector<int32_t>& live_tokens,
                        EmitFn&& on_emit, StopCheckFn&& is_stopped, bool* out_hit_eos = nullptr) {
    const int V = (int) E.m->hp.vocab_unpadded;
    const bool sample_on = smp.needs_logits();
    int32_t next_tok = first_tok;
    int emitted = 0;
    std::vector<float> lg;
    std::vector<int32_t> d_toks, vtoks, t_picks;
    std::vector<std::vector<float>> d_logits;
    std::vector<SamplingDist> q_dists;
    if (out_hit_eos) *out_hit_eos = false;

    auto feed_single = [&](int32_t tok) {
        const int pos_pre = (int) E.pos;
        next_tok = E.forward(&tok, 1, sample_on, &lg, false, dspark != nullptr, 0, nullptr);
        live_tokens.push_back(tok);
        if (dspark) dspark->inject(E, pos_pre, 1);
        if (sample_on) next_tok = sample_logits(lg, hist, smp, rng);
    };

    while (emitted < limit && !is_stopped()) {
        if (!on_emit(next_tok)) {
            if (out_hit_eos) *out_hit_eos = true;
            ++emitted;
            break;
        }
        ++emitted;
        const int rem = limit - emitted;
        if (rem <= 0 || is_stopped()) break;

        if (!dspark || rem < 2 || draft_max < 1) {
            feed_single(next_tok);
            if (E.any_hybrid() && (emitted % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 6);
            continue;
        }

        const int K = std::min(draft_max, rem - 1);
        const int32_t anchor = next_tok;
        if (sample_on) {
            q_dists.resize((size_t) K);
            dspark->draft(E, anchor, K, d_toks, &d_logits, [&](int step_i, std::vector<float>& row_lg) {
                std::vector<int32_t> tmp_hist = hist;
                tmp_hist.insert(tmp_hist.end(), d_toks.begin(), d_toks.begin() + step_i);
                q_dists[(size_t) step_i] = build_sampling_dist(row_lg.data(), V, tmp_hist, smp);
                return sample_from_dist(q_dists[(size_t) step_i], rng);
            });
        } else {
            dspark->draft(E, anchor, K, d_toks, nullptr, [&](int, std::vector<float>&) -> int32_t { return 0; });
        }

        if (is_stopped()) break;   // no target state was advanced by drafting

        vtoks.resize((size_t) (1 + K));
        vtoks[0] = anchor;
        for (int i = 0; i < K; ++i) vtoks[(size_t) (1 + i)] = d_toks[(size_t) i];

        const int pos_pre = (int) E.pos;
        lg.clear();
        E.forward(vtoks.data(), 1 + K, sample_on, &lg, sample_on, true, K, &t_picks);

        int m = 0;
        bool hit_eos = false;
        if (!sample_on) {
            int matching = 0;
            while (matching < K && t_picks[(size_t) matching] == d_toks[(size_t) matching]) ++matching;
            for (int j = 0; j < matching && !is_stopped(); ++j) {
                ++m;
                if (!on_emit(d_toks[(size_t) j])) { hit_eos = true; break; }
            }
            if (!hit_eos && !is_stopped()) next_tok = t_picks[(size_t) m];
        } else {
            bool rejected = false;
            for (int r = 0; r < K && !is_stopped(); ++r) {
                const SamplingDist p_r = build_sampling_dist(lg.data() + (size_t) r * V, V, hist, smp);
                const double p_x = p_r.prob_of(d_toks[(size_t) r]);
                const double q_x = q_dists[(size_t) r].prob_of(d_toks[(size_t) r]);
                const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                if (u * (double) q_x <= (double) p_x) {
                    ++m;
                    if (!on_emit(d_toks[(size_t) r])) {
                        hit_eos = true;
                        break;
                    }
                } else {
                    next_tok = sample_residual_dist(p_r, q_dists[(size_t) r], rng);
                    rejected = true;
                    break;
                }
            }
            if (!rejected && !hit_eos && !is_stopped()) {
                const SamplingDist p_last = build_sampling_dist(lg.data() + (size_t) K * V, V, hist, smp);
                next_tok = sample_from_dist(p_last, rng);
            }
        }

        emitted += m;
        dspark->dump_verify_logits(pos_pre, K, V, lg, m);
        E.commit_verify(m, K, pos_pre);
        for (int i = 0; i < 1 + m; ++i) live_tokens.push_back(vtoks[(size_t) i]);
        dspark->n_accepted += m;
        dspark->inject(E, pos_pre, 1 + m);
        if (is_stopped()) break;
        if (hit_eos) {
            if (out_hit_eos) *out_hit_eos = true;
            break;
        }
        if (E.any_hybrid() && (emitted % 8 < (1 + m))) E.rebalance(E.core_all_gpu ? 4 : 6);
    }
}


// Opt-in diagnostic only. The prefix is reconstructed with the original prompt chunking
// followed by the requested singleton tail. The draft is loaded but never runs draft/inject.
int verify_row0_oracle(Engine& E, const std::vector<int32_t>& prefix, const std::vector<int32_t>& tail,
                      int force_t1, int chunk) {
    if (!E.dspark || E.verify_max_t < 8 || E.tmax < 8 || tail.size() != 8 ||
        force_t1 < 0 || (size_t) force_t1 > prefix.size() || chunk <= 0 ||
        prefix.size() + 8 > (size_t) E.ctx) {
        std::fprintf(stderr, "strata-inkling2: row0 oracle needs --draft, --spec 7, eight --verify-row0-oracle IDs and a valid --oracle-force-t1 extent\n");
        return 2;
    }
    const int V = (int) E.m->hp.vocab_unpadded;
    for (int32_t tok : tail) if (tok < 0 || tok >= V) {
        std::fprintf(stderr, "strata-inkling2: row0 oracle token outside vocabulary\n"); return 2;
    }
    E.reset_state();
    const size_t prompt_end = prefix.size() - (size_t) force_t1;
    for (size_t off = 0; off < prompt_end; off += (size_t) chunk) {
        const int n = (int) std::min<size_t>((size_t) chunk, prompt_end - off);
        E.forward(prefix.data() + off, n);
    }
    for (size_t off = prompt_end; off < prefix.size(); ++off) E.forward(prefix.data() + off, 1);
    const auto checkpoint = E.save_checkpoint(prefix);
    auto restore = [&] {
        E.restore_checkpoint(checkpoint);
        { DeviceScope owner(E.device); CK(cudaStreamSynchronize(E.st)); }
        if (E.split_peer) { DeviceScope owner(E.split_peer->device); CK(cudaStreamSynchronize(E.split_peer->st)); }
    };
    auto number = [](double x) {
        if (!std::isfinite(x)) return std::string("null");
        char text[64]; std::snprintf(text, sizeof text, "%.17g", x); return std::string(text);
    };
    auto compare = [&](const char* kind, int T, const std::vector<float>& row, const std::vector<float>& reference) {
        double max_abs = 0, sum = 0, diff_sq = 0, ref_sq = 0;
        int nonfinite = 0, best = 0, second = -1;
        for (int i = 0; i < V; ++i) {
            const double a = row[(size_t) i], b = reference[(size_t) i];
            if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
            const double d = a - b;
            max_abs = std::max(max_abs, std::fabs(d)); sum += d; diff_sq += d * d; ref_sq += b * b;
            if (i != best) {
                if (a > row[(size_t) best]) { second = best; best = i; }
                else if (second < 0 || a > row[(size_t) second]) second = i;
            }
        }
        const double l_a = V > 150759 ? row[150759] : NAN;
        const double l_b = V > 150078 ? row[150078] : NAN;
        std::printf("{\"diagnostic\":\"verify_row0\",\"case\":\"%s\",\"T\":%d,\"pos\":%lld,\"argmax\":%d,\"runner_up\":%d,"
                    "\"argmax_margin\":%s,\"max_abs\":%s,\"mean_delta\":%s,\"rms_delta\":%s,\"rel_l2\":%s,\"nonfinite\":%d,"
                    "\"logit_150759\":%s,\"logit_150078\":%s,\"probe_margin\":%s}\n",
                    kind, T, (long long) prefix.size(), best, second,
                    number(second >= 0 ? (double) row[(size_t) best] - row[(size_t) second] : NAN).c_str(),
                    number(max_abs).c_str(), number(sum / V).c_str(), number(std::sqrt(diff_sq / V)).c_str(),
                    number(std::sqrt(diff_sq / (ref_sq + 1e-30))).c_str(), nonfinite,
                    number(l_a).c_str(), number(l_b).c_str(), number(l_a - l_b).c_str());
        std::fflush(stdout);
    };
    std::printf("{\"diagnostic\":\"verify_row0_setup\",\"prompt_chunk_rows\":%zu,\"forced_t1_rows\":%d,\"chunk\":%d,"
                "\"anchor\":%d,\"draft_called\":false,\"inject_called\":false,\"commit_verify_called\":false}\n",
                prompt_end, force_t1, chunk, tail[0]);
    std::vector<float> baseline, batch8;
    for (int T : {1, 2, 4, 8}) {
        restore();
        std::vector<float> logits;
        // Trace/backup run exactly as in the verifier; no commit follows. Checkpoint restore
        // supplies the same pre-anchor state independently for every batch shape.
        E.forward(tail.data(), T, true, &logits, true, true, T - 1);
        logits.resize((size_t) V);
        if (T == 1) baseline = logits;
        if (T == 8) batch8 = logits;
        compare("batch_vs_T1", T, logits, baseline);
    }
    auto alternate = tail;
    for (int i = 1; i < 8; ++i) alternate[(size_t) i] = (int32_t) ((tail[(size_t) i] + 7919ll * i) % V);
    restore();
    std::vector<float> logits;
    E.forward(alternate.data(), 8, true, &logits, true, true, 7);
    logits.resize((size_t) V);
    compare("future_tail_vs_T8", 8, logits, batch8);
    restore();
    return 0;
}


}  // namespace

int run(int argc, char** argv) {
    std::string model_path;
    std::string draft_path;
    std::string oracle_tail;
    int oracle_force_t1 = 0;
    int draft_max = 3;
    std::string tokens_str = "1,2";
    std::string dump_logits_path;
    int max_ctx = 65536;
    int chunk = 256;
    int max_new = 32;
    int cpu_threads = 0;
    int layer_split_device = -1;
    int expert_replica_device = -1, expert_dense_device = -1;
    int expert_hot_mib = 4096;
    bool expert_replica_async = false;
    bool split_core_graph = false;
    bool split_outer_graph = false;
    bool batch_qkv_control_t1 = false;
    double expert_vram = -1.0;
    bool serve = false;
    bool bench = false;
    bool kv_q8 = true;
    bool profile = false;
    bool profile_detail = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_arg = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-inkling2: missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--native" || a == "--model") model_path = next_arg();
        else if (a == "--verify-row0-oracle") oracle_tail = next_arg();
        else if (a == "--oracle-force-t1") oracle_force_t1 = std::atoi(next_arg());
        else if (a == "--draft" || a == "--draft-model") draft_path = next_arg();
        else if (a == "--spec" || a == "--draft-max") draft_max = std::clamp(std::atoi(next_arg()), 1, 7);
        else if (a == "--max-context") max_ctx = std::atoi(next_arg());
        else if (a == "--chunk" || a == "--prefill") chunk = std::atoi(next_arg());
        else if (a == "--max-new" || a == "-n") max_new = std::atoi(next_arg());
        else if (a == "--tokens") tokens_str = next_arg();
        else if (a == "--expert-replica-async") expert_replica_async = true;
        else if (a == "--expert-replica-device" || a == "--expert-dense-device" || a == "--expert-hot-mib") {
            const char* text = next_arg(); char* tail = nullptr;
            const long value = std::strtol(text, &tail, 10);
            const long limit = a == "--expert-hot-mib" ? 1048576 : 63;
            if (!text[0] || *tail || value < 0 || value > limit) {
                std::fprintf(stderr, "strata-inkling2: invalid value for %s\n", a.c_str()); return 2;
            }
            if (a == "--expert-replica-device") expert_replica_device = (int) value;
            else if (a == "--expert-dense-device") expert_dense_device = (int) value;
            else expert_hot_mib = (int) value;
        }
        else if (a == "--expert-vram") expert_vram = std::atof(next_arg());
        else if (a == "--cpu-threads" || a == "--pool-workers") cpu_threads = std::atoi(next_arg());
        else if (a == "--dump-logits") dump_logits_path = next_arg();
        else if (a == "--layer-split-device") {
            const char* text = next_arg();
            char* tail = nullptr;
            const long value = std::strtol(text, &tail, 10);
            if (!text[0] || *tail || value < 0 || value > 63) {
                std::fprintf(stderr, "strata-inkling2: --layer-split-device requires a CUDA device index (0..63)\n");
                return 2;
            }
            layer_split_device = (int) value;
        }
        else if (a == "--split-core-graph") split_core_graph = true;
        else if (a == "--split-outer-graph") split_outer_graph = true;
        else if (a == "--batch-qkv-control-t1") batch_qkv_control_t1 = true;
        else if (a == "--kv-f16") kv_q8 = false;
        else if (a == "--kv-q8") kv_q8 = true;
        else if (a == "--serve") serve = true;
        else if (a == "--bench") bench = true;
        else if (a == "--profile") profile = true;
        else if (a == "--profile-detail") { profile = true; profile_detail = true; }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "usage: strata-inkling2 --model <file.gguf> [--max-context N] [--kv-q8|--kv-f16] [--tokens IDS] [--max-new N] [--dump-logits PATH] [--layer-split-device N [--split-core-graph] [--split-outer-graph]] [--batch-qkv-control-t1] [--expert-dense-device N --expert-replica-device N --expert-hot-mib MiB [--expert-replica-async]] [--draft DRAFT.gguf --spec K] [--bench] [--serve]\n");
        return 2;
    }
    if (split_core_graph && layer_split_device < 0) {
        std::fprintf(stderr, "strata-inkling2: --split-core-graph requires --layer-split-device N\n");
        return 2;
    }
    if (split_outer_graph && layer_split_device < 0) {
        std::fprintf(stderr, "strata-inkling2: --split-outer-graph requires --layer-split-device N\n"); return 2;
    }

    if (oracle_tail.empty() && oracle_force_t1 != 0) {
        std::fprintf(stderr, "strata-inkling2: --oracle-force-t1 needs --verify-row0-oracle\n"); return 2;
    }
    if (!oracle_tail.empty() && serve) {
        std::fprintf(stderr, "strata-inkling2: --verify-row0-oracle is a standalone CLI diagnostic, not a serve mode\n"); return 2;
    }

    if (expert_replica_async && expert_replica_device < 0) {
        std::fprintf(stderr, "strata-inkling2: --expert-replica-async requires expert replica placement\n"); return 2;
    }
    if (expert_replica_device >= 0 || expert_dense_device >= 0) {
        int count = 0; CK(cudaGetDeviceCount(&count));
        if (expert_replica_device < 0 || expert_dense_device < 0 || expert_replica_device == expert_dense_device ||
            expert_replica_device >= count || expert_dense_device >= count || layer_split_device >= 0 ||
            !draft_path.empty() || max_ctx != 262144 || !kv_q8 || chunk < 1 || chunk > 64 || expert_vram >= 0 ||
            split_core_graph || split_outer_graph || !oracle_tail.empty()) {
            std::fprintf(stderr, "strata-inkling2: expert replica requires distinct available --expert-dense-device and --expert-replica-device, --max-context 262144 --kv-q8 --chunk 1..64 (operational:64); draft/MTP, layer split, split graphs, oracle and --expert-vram are incompatible\n"); return 2;
        }
        CK(cudaSetDevice(expert_dense_device));
    }

    Model m;
    CK(cudaGetDevice(&m.main_device));
    m.split_device = layer_split_device;
    if (layer_split_device >= 0) {
        int device_count = 0;
        CK(cudaGetDeviceCount(&device_count));
        if (layer_split_device >= device_count || max_ctx != 262144 || !kv_q8) {
            std::fprintf(stderr, "strata-inkling2: layer split requires an available CUDA device and --max-context 262144 --kv-q8\n");
            return 2;
        }
    }
    std::string err;
    if (!load_model(model_path, m, err)) {
        std::fprintf(stderr, "strata-inkling2: load failed: %s\n", err.c_str());
        return 1;
    }

    std::unique_ptr<DsparkModel> dm;
    std::unique_ptr<DsparkEngine> dspark;
    if (!draft_path.empty()) {
        dm = std::make_unique<DsparkModel>();
        dm->device = m.main_device;
        if (!load_dspark_model(draft_path, m.hp, *dm, err)) {
            std::fprintf(stderr, "strata-inkling2: draft load failed: %s\n", err.c_str()); return 1;
        }
    }
    const int runtime_t = dm ? std::max(chunk, dm->hp.block_size) : chunk;
    const int verify_t = dm ? draft_max + 1 : 1;
    Engine E;
    E.device = m.main_device;
    E.role = layer_split_device >= 0 ? Engine::Role::Outer : Engine::Role::All;
    E.split_outer_graph = split_outer_graph;
    E.batch_qkv_control_t1 = batch_qkv_control_t1;
    E.outer_graph_profile_block = profile || profile_detail;
    if (!E.init(m, max_ctx, runtime_t, kv_q8, err, verify_t, dm && dm->has_quant_dense, dm ? dm->hp.n_ff : 0)) {
        std::fprintf(stderr, "strata-inkling2: engine init failed: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<Engine> split_core;
    if (layer_split_device >= 0) {
        if (m.hp.n_layer != 48 || m.hp.n_phys != 24 || m.hp.core_start != 6 || m.hp.core_end != 18 ||
            m.hp.num_loops != 3 || m.hp.attn_res_block_size != 6 || m.hp.n_embd != 2304 || m.hp.n_ff != 1152) {
            std::fprintf(stderr, "strata-inkling2: layer split requires Inkling2's 48/24-layer, 3-loop 6..17 topology\n");
            return 2;
        }
        split_core = std::make_unique<Engine>();
        split_core->device = layer_split_device;
        split_core->role = Engine::Role::Core;
        split_core->split_core_graph = split_core_graph;
        split_core->batch_qkv_control_t1 = batch_qkv_control_t1;
        if (!split_core->init(m, max_ctx, runtime_t, kv_q8, err, verify_t, false)) {
            std::fprintf(stderr, "strata-inkling2: split core init failed: %s\n", err.c_str()); return 1;
        }
        E.split_peer = split_core.get();
        split_core->split_peer = &E;
        {
            DeviceScope owner(E.device);
            CK(cudaHostAlloc(&E.split_handoff, (size_t) 4 * E.tmax * m.hp.n_embd * sizeof(float), cudaHostAllocPortable));
        }
        std::fprintf(stderr, "strata-inkling2: layer split main_device=%d virtual=0..5,42..47+head core_device=%d virtual=6..41 (physical=6..17 x3), context=262144 KV=Q8, host handoff=18/36 KiB per token, core_graph=%s\n",
                     E.device, split_core->device, split_core_graph ? "enabled" : "disabled (eager)");
    }
    if (dm) {
        dspark = std::make_unique<DsparkEngine>();
        dspark->init(*dm, max_ctx, E.tmax);
        E.enable_dspark(dspark.get());
        if (split_core) split_core->enable_dspark(dspark.get());
        E.reset_state();
    }
    if (!(expert_replica_device >= 0
              ? place_replica_experts(m, E, expert_replica_device, (size_t) expert_hot_mib << 20, expert_replica_async, err)
              : place_experts(m, E, expert_vram, cpu_threads, err)) ||
        (split_core && !place_experts(m, *split_core, expert_vram, cpu_threads, err))) {
        std::fprintf(stderr, "strata-inkling2: expert placement failed: %s\n", err.c_str());
        return 1;
    }
    if (split_core) {
        if (split_outer_graph)
            std::fprintf(stderr, "strata-inkling2: split outer graph requested=1 eligible=%d (T=1, no draft/capture/trace/profile, all outer experts resident; head excluded)\n",
                         (int) E.outer_graph_eligible());
        std::fprintf(stderr, "strata-inkling2: split core graph requested=%d eligible=%d (T=1, fully resident core, profile-detail off)\n",
                     (int) split_core_graph, (int) (split_core_graph && split_core->core_all_gpu && !profile_detail));
        size_t total = 0;
        { DeviceScope owner(E.device); CK(cudaMemGetInfo(&E.vram_free_after_place, &total)); }
        { DeviceScope owner(split_core->device); CK(cudaMemGetInfo(&split_core->vram_free_after_place, &total)); }
        // These fields describe the public model as a whole; each runtime's live counters
        // and cache ownership remain separate.
        E.placed_slots += split_core->placed_slots;
        E.placed_bytes += split_core->placed_bytes;
        E.pool_workers += split_core->pool_workers;
        std::fprintf(stderr, "strata-inkling2: split VRAM free main=%.0f MiB core=%.0f MiB; %d/%lld experts resident\n",
                     (double) E.vram_free_after_place / (1 << 20), (double) split_core->vram_free_after_place / (1 << 20),
                     E.placed_slots, (long long) m.hp.n_phys * m.hp.n_expert);
    }

    if (serve) {
        int32_t eos_id = 200006;
        if (const auto* ev = m.gguf->meta().get("tokenizer.ggml.eos_token_id")) {
            if (ev->is_num()) eos_id = (int32_t) ev->u;
        }
        std::mutex q_mu;
        std::condition_variable q_cv;
        std::deque<std::string> q_lines;
        bool stdin_eof = false;
        std::atomic<bool> stop_req{false};

        std::thread reader([&] {
            std::string line;
            char chunk_buf[65536];
            while (std::fgets(chunk_buf, sizeof chunk_buf, stdin)) {
                line += chunk_buf;
                if (!line.empty() && line.back() != '\n' && !std::feof(stdin)) continue;
                while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
                if (line == "STOP") {
                    stop_req.store(true, std::memory_order_release);
                    line.clear();
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lk(q_mu);
                    q_lines.push_back(std::move(line));
                }
                q_cv.notify_one();
                line.clear();
            }
            {
                std::lock_guard<std::mutex> lk(q_mu);
                stdin_eof = true;
            }
            q_cv.notify_one();
        });

        std::printf("INFO engine=0.2.17-inkling2 context=%d kv=%s kv_resident=%d expert_slots=%d experts=%lld expert_cache_mib=%lld vram_free_mib=%lld pool_workers=%d dspark=%s draft_max=%d\n",
                    max_ctx, kv_q8 ? "int8" : "fp16", max_ctx,
                    E.placed_slots, (long long) m.hp.n_phys * m.hp.n_expert,
                    (long long) (E.placed_bytes >> 20), (long long) (E.vram_free_after_place >> 20),
                    E.pool_workers, dspark ? "on" : "off", dspark ? draft_max : 0);
        std::printf("READY %d stop\n", max_ctx);
        std::fflush(stdout);

        std::vector<float> lg;
        std::vector<int32_t> hist;
        std::vector<int32_t> live_tokens;
        std::vector<Engine::StateCheckpoint> snaps;
        constexpr size_t kMaxSnaps = 8;
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);

        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&] { return !q_lines.empty() || stdin_eof; });
                if (q_lines.empty()) break;
                line = std::move(q_lines.front());
                q_lines.pop_front();
            }
            if (line == "QUIT" || line == "EXIT") break;
            if (line == "RESET" && E.replica) {
                E.reset_state();
                E.reset_hybrid_stats();
                live_tokens.clear(); snaps.clear(); hist.clear();
                std::printf("RESET_OK\n"); std::fflush(stdout);
                continue;
            }
            if (line.rfind("GEN ", 0) == 0 || line == "GEN") {
                stop_req.store(false, std::memory_order_release);
                std::istringstream iss(line.size() > 4 ? line.substr(4) : "");
                int req_new = 0;
                iss >> req_new;
                SamplingParams smp;
                std::vector<int32_t> prompt;
                std::string tok;
                while (iss >> tok) {
                    const size_t eq = tok.find('=');
                    if (eq != std::string::npos) {
                        const std::string k = tok.substr(0, eq);
                        const std::string v = tok.substr(eq + 1);
                        if (k == "temperature") smp.temp = std::strtof(v.c_str(), nullptr);
                        else if (k == "top_p") smp.top_p = std::strtof(v.c_str(), nullptr);
                        else if (k == "top_k") smp.top_k = std::atoi(v.c_str());
                        else if (k == "min_p") smp.min_p = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_repeat") smp.rep_pen = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_freq") smp.freq_pen = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_present") smp.pres_pen = std::strtof(v.c_str(), nullptr);
                        else if (k == "penalty_last_n") smp.pen_last_n = std::atoi(v.c_str());
                        else if (k == "seed") smp.seed = (uint64_t) std::strtoull(v.c_str(), nullptr, 10);
                        continue;
                    }
                    for (int32_t id : parse_id_list(tok)) prompt.push_back(id);
                }
                if (smp.seed > 0) rng.seed(smp.seed);
                if (prompt.empty()) {
                    std::printf("DONE 0 0 0 0 stop\n");
                    std::fflush(stdout);
                    continue;
                }
                if ((int) prompt.size() >= max_ctx) {
                    std::printf("ERR prompt (%zu) exceeds max_context (%d)\n", prompt.size(), max_ctx);
                    std::fflush(stdout);
                    continue;
                }

                // Prefix / KV cache reuse along live_tokens & saved SWA/conv checkpoints
                const size_t max_possible = prompt.size() - 1;
                size_t lcp_live = 0;
                const size_t lim_live = std::min(live_tokens.size(), max_possible);
                while (lcp_live < lim_live && live_tokens[lcp_live] == prompt[lcp_live]) ++lcp_live;

                size_t reuse = 0;
                if (lcp_live == live_tokens.size()) {
                    reuse = lcp_live;
                } else {
                    int best_idx = -1;
                    for (int si = (int) snaps.size() - 1; si >= 0; --si) {
                        const auto& stoks = snaps[(size_t) si].tokens;
                        if (stoks.size() <= lcp_live &&
                            std::equal(stoks.begin(), stoks.end(), prompt.begin())) {
                            best_idx = si;
                            break;
                        }
                    }
                    if (best_idx >= 0) {
                        E.restore_checkpoint(snaps[(size_t) best_idx]);
                        live_tokens = snaps[(size_t) best_idx].tokens;
                        snaps.resize((size_t) best_idx + 1);
                        reuse = live_tokens.size();
                    } else {
                        E.reset_state();
                        live_tokens.clear();
                        snaps.clear();
                        reuse = 0;
                    }
                }
                std::printf("RESUME %zu\n", reuse);
                std::fflush(stdout);

                const auto t_pp0 = Clock::now();
                E.reset_hybrid_stats();
                if (dspark) { dspark->n_draft_calls = dspark->n_drafted = dspark->n_accepted = 0; dspark->draft_ms = dspark->inject_ms = 0; }
                hist = prompt;
                const bool sample_on = smp.needs_logits();
                int32_t next_tok = 0;
                const size_t fresh = prompt.size() - reuse;
                for (size_t off = reuse; off < prompt.size() && !stop_req.load(std::memory_order_acquire); off += (size_t) chunk) {
                    const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
                    const bool is_last = (off + (size_t) tc == prompt.size());
                    const int p0 = (int) E.pos;
                    next_tok = E.forward(prompt.data() + off, tc, is_last && sample_on, &lg, false, dspark != nullptr);
                    if (dspark) dspark->inject(E, p0, tc);
                    live_tokens.insert(live_tokens.end(), prompt.begin() + (ptrdiff_t) off, prompt.begin() + (ptrdiff_t) (off + (size_t) tc));
                    if (E.any_hybrid()) E.rebalance(8);
                    const double el_ms = ms_since(t_pp0);
                    const double done_fresh = (double) (off + (size_t) tc - reuse);
                    const double pp_tps_now = done_fresh / std::max(1e-6, el_ms / 1000.0);
                    std::printf("PP %zu %zu %.0f %.1f\n", off + (size_t) tc, prompt.size(), el_ms, pp_tps_now);
                    std::fflush(stdout);
                    if (!is_last && !stop_req.load(std::memory_order_acquire)) {
                        const size_t last_cp = snaps.empty() ? 0 : snaps.back().tokens.size();
                        if (live_tokens.size() - last_cp >= 2048) {
                            snaps.push_back(E.save_checkpoint(live_tokens));
                            if (snaps.size() > kMaxSnaps) snaps.erase(snaps.begin());
                        }
                    }
                }
                if (!stop_req.load(std::memory_order_acquire) && live_tokens.size() == prompt.size()) {
                    if (snaps.empty() || snaps.back().tokens.size() != live_tokens.size()) {
                        snaps.push_back(E.save_checkpoint(live_tokens));
                        if (snaps.size() > kMaxSnaps) snaps.erase(snaps.begin());
                    }
                }
                const double pp_ms = ms_since(t_pp0);
                if (sample_on && !stop_req.load(std::memory_order_acquire)) {
                    next_tok = sample_logits(lg, hist, smp, rng);
                }

                const auto t_tg0 = Clock::now();
                int emitted = 0;
                const char* finish = "length";
                const int avail = std::max(0, max_ctx - (int) prompt.size());
                const int limit = req_new > 0 ? std::min(req_new, avail) : avail;
                if (stop_req.load(std::memory_order_acquire)) {
                    finish = "cancel";
                } else if (dspark) {
                    bool hit_eos = false;
                    decode_tokens_loop(E, dspark.get(), draft_max, next_tok, limit, smp, rng, hist, live_tokens,
                        [&](int32_t tok) -> bool {
                            std::printf("T %d\n", (int) tok); std::fflush(stdout);
                            ++emitted; hist.push_back(tok);
                            return tok != eos_id && tok != 199999;
                        }, [&] { return stop_req.load(std::memory_order_acquire); }, &hit_eos);
                    if (stop_req.load(std::memory_order_acquire)) finish = "cancel";
                    else if (hit_eos) finish = "stop";
                } else {
                    for (int step = 0; step < limit; ++step) {
                        if (stop_req.load(std::memory_order_acquire)) {
                            finish = "cancel";
                            break;
                        }
                        std::printf("T %d\n", (int) next_tok);
                        std::fflush(stdout);
                        ++emitted;
                        hist.push_back(next_tok);
                        if (next_tok == eos_id || next_tok == 199999) {
                            finish = "stop";
                            break;
                        }
                        if (step + 1 < limit) {
                            const int32_t fed = next_tok;
                            next_tok = E.forward(&fed, 1, sample_on, &lg);
                            live_tokens.push_back(fed);
                            if (sample_on) next_tok = sample_logits(lg, hist, smp, rng);
                            if (E.any_hybrid() && (step % 8 == 7)) E.rebalance(E.core_all_gpu ? 4 : 6);
                        }
                    }
                }
                const double tg_ms = ms_since(t_tg0);
                const long long lookups = (long long) E.total_hy_gpu() + (long long) E.total_hy_cpu();
                std::printf("DONE %d %zu %.2f %.2f %s %lld %lld %zu %lld %lld 0 0 0.0 %zu\n",
                            emitted, prompt.size(), pp_ms, tg_ms, finish,
                            dspark ? (long long) dspark->n_accepted : 0LL, dspark ? (long long) dspark->n_drafted : 0LL, reuse, (long long) E.total_hy_gpu(), lookups, fresh);
                std::fflush(stdout);
                E.print_replica_stats("DONE");
            }
        }
        if (reader.joinable()) reader.join();
        return 0;
    }

    std::vector<int32_t> prompt = parse_id_list(tokens_str);
    if (prompt.empty()) prompt = {1};
    if (!oracle_tail.empty()) return verify_row0_oracle(E, prompt, parse_id_list(oracle_tail), oracle_force_t1, chunk);

    if (bench) {
        // Warmup 1 pass
        E.reset_state();
        int32_t w = 0;
        for (size_t off = 0; off < prompt.size(); off += (size_t) chunk) {
            const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
            const int p0 = (int) E.pos;
            w = E.forward(prompt.data() + off, tc, false, nullptr, false, dspark != nullptr);
            if (dspark) dspark->inject(E, p0, tc);
        }
        if (max_new > 0) E.forward(&w, 1);
        E.reset_state();
        E.reset_hybrid_stats();
        if (dspark) { dspark->n_draft_calls = dspark->n_drafted = dspark->n_accepted = 0; dspark->draft_ms = dspark->inject_ms = 0; }
    }

    E.prof = profile;
    E.prof_detail = profile_detail;
    E.reset_prof();
    std::vector<float> dumped;
    const auto t_pp0 = Clock::now();
    int32_t next_tok = 0;
    for (size_t off = 0; off < prompt.size(); off += (size_t) chunk) {
        const int tc = (int) std::min<size_t>((size_t) chunk, prompt.size() - off);
        const bool is_last = (off + (size_t) tc == prompt.size());
        const int p0 = (int) E.pos;
        next_tok = E.forward(prompt.data() + off, tc, is_last && !dump_logits_path.empty(), &dumped, false, dspark != nullptr);
        if (dspark) dspark->inject(E, p0, tc);
        const auto tr0 = Clock::now();
        if (E.any_hybrid()) E.rebalance(8);
        if (profile) { CK(cudaStreamSynchronize(E.st)); E.p_rebal_ms += ms_since(tr0); }
    }
    const double pp_ms = ms_since(t_pp0);
    if (profile) { E.print_prof("pp"); E.reset_prof(); }

    if (!dump_logits_path.empty() && !dumped.empty()) {
        std::FILE* f = std::fopen(dump_logits_path.c_str(), "wb");
        if (f) {
            std::fwrite(dumped.data(), sizeof(float), dumped.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "strata-inkling2: wrote %zu logits to %s\n", dumped.size(), dump_logits_path.c_str());
        }
    }

    std::vector<int32_t> gen;
    gen.reserve((size_t) max_new);
    if (max_new > 0 && !dspark) gen.push_back(next_tok);

    const auto t_tg0 = Clock::now();
    if (dspark) {
        std::vector<int32_t> cli_hist = prompt, cli_live = prompt;
        SamplingParams greedy;
        std::mt19937_64 cli_rng(0x9e3779b97f4a7c15ULL);
        decode_tokens_loop(E, dspark.get(), draft_max, next_tok, max_new, greedy, cli_rng, cli_hist, cli_live,
            [&](int32_t tok) -> bool { gen.push_back(tok); cli_hist.push_back(tok); return true; }, [] { return false; });
    } else {
        for (int step = 1; step < max_new; ++step) {
            next_tok = E.forward(&next_tok, 1);
            gen.push_back(next_tok);
            const auto tr0 = Clock::now();
            if (E.any_hybrid() && (step % 8 == 0)) E.rebalance(E.core_all_gpu ? 4 : 6);
            if (profile) { CK(cudaStreamSynchronize(E.st)); E.p_rebal_ms += ms_since(tr0); }
        }
    }
    const double tg_ms = ms_since(t_tg0);
    E.print_replica_stats("DONE");
    if (profile) E.print_prof("tg");
    if (dspark)
        std::fprintf(stderr, "strata-inkling2: dspark accepted=%lld offered=%lld draft_calls=%lld draft_ms=%.1f inject_ms=%.1f\n",
                     (long long) dspark->n_accepted, (long long) dspark->n_drafted, (long long) dspark->n_draft_calls,
                     dspark->draft_ms, dspark->inject_ms);

    std::printf("tokens:");
    for (int32_t id : gen) std::printf(" %d", (int) id);
    std::printf("\n");

    const double pp_tps = (double) prompt.size() / (pp_ms / 1000.0);
    const double tg_tps = max_new > 1 ? (double) (max_new - 1) / (tg_ms / 1000.0) : 0.0;
    std::fprintf(stderr, "strata-inkling2: pp %zu tok in %.1f ms (%.2f t/s) | tg %d tok in %.1f ms (%.2f t/s) | hits=%lld misses=%lld cpu_ms=%.1f\n",
                 prompt.size(), pp_ms, pp_tps, std::max(0, max_new - 1), tg_ms, tg_tps,
                 (long long) E.total_hy_gpu(), (long long) E.total_hy_cpu(), E.total_hy_cpu_ms());
    return 0;
}

}  // namespace strata::inkling2

