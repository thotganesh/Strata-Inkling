# Inkling-2-Mini Q8: cache esperti con ribilanciamento bulk dopo il prefill
param(
  [string]$Model = "C:\models\inkling\Inkling-2-Mini-UD-Q8_K_XL.gguf",
  [int]$MaxContext = 4096,
  [int]$Chunk = 512,
  [int]$MaxNew = 256,
  [string]$TokensFile = "C:\models\inkling\pw0.txt",
  [switch]$Serve
)
Set-Location C:\Strata-Inkling
$env:STRATA_MAX_PIN_MB = 16500
$env:STRATA_REBAL_BULK_PER_LAYER = 16
$env:STRATA_REBAL_BULK_SWAPS = 800
$exe = ".\build-win\strata-inkling2.exe"
if ($Serve) {
  & $exe --model $Model --kv-q8 --max-context $MaxContext --chunk $Chunk --serve
} else {
  $tt = Get-Content $TokensFile -Raw
  & $exe --model $Model --kv-q8 --max-context $MaxContext --chunk $Chunk --max-new $MaxNew --tokens $tt --bench
}
