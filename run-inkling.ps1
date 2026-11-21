# Inkling-2-Mini Q4 abliterated su Strata (fork): serve con web UI locale. Default 32k di contesto.
# Uso:  .\run-inkling.ps1            -> http://127.0.0.1:8085
#       .\run-inkling.ps1 -Context 65536 -Chunk 256
#       .\run-inkling.ps1 -Bench     -> bench senza server (usa $TokensFile)
param(
  [string]$Model = "C:\models\inkling\Inkling-2-Mini-UD-Q4_K_XL-Abliterated.gguf",
  [int]$MaxContext = 32768,
  [int]$Chunk = 512,
  [int]$Port = 8085,
  [string]$TokenizerDir = "C:\models\inkling\tokenizer\tokenizer",
  [string]$TokensFile = "C:\models\inkling\pw0.txt",
  [int]$MaxNew = 128,
  [switch]$Bench
)
$root = $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "build-win\strata-inkling2.exe"
$env_cfg = [ordered]@{
  STRATA_MAX_PIN_MB           = "16500"
  STRATA_REBAL_BULK_PER_LAYER = "16"
  STRATA_REBAL_BULK_SWAPS     = "800"
}
if ($Bench) {
  foreach ($k in $env_cfg.Keys) { Set-Item "Env:$k" $env_cfg[$k] }
  $tt = Get-Content $TokensFile -Raw
  & $exe --model $Model --kv-q8 --max-context $MaxContext --chunk $Chunk --max-new $MaxNew --tokens $tt --bench
  return
}
$cfg = [ordered]@{
  arch           = "inkling2"
  model_name     = [IO.Path]::GetFileNameWithoutExtension($Model)
  exe            = $exe
  args           = @("--model",$Model,"--kv-q8","--max-context","$MaxContext","--chunk","$Chunk","--serve")
  tokenizer      = $TokenizerDir
  env            = $env_cfg
  fit_max_tokens = $true
  no_api_key     = $true
}
$cfgPath = Join-Path $root "inkling-run.json"
$cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $cfgPath
"Inkling: contesto $MaxContext, chunk $Chunk -> http://127.0.0.1:$Port (attendi ~40 s)"
& (Join-Path $root "terminal\bin\strata.cmd") serve --config $cfgPath --host 127.0.0.1 --port $Port

