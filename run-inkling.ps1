# Inkling-2-Mini Q4 abliterated su Strata (fork): serve con web UI e API OpenAI.
# Uso:  .\run-inkling.ps1                          -> solo questo PC: http://127.0.0.1:8085
#       .\run-inkling.ps1 -Lan -ApiKey "chiave"    -> anche altri dispositivi / Open WebUI
#       .\run-inkling.ps1 -MaxContext 65536 -Chunk 256
#       .\run-inkling.ps1 -Bench                   -> bench senza server
param(
  [string]$Model = "C:\models\inkling\Inkling-2-Mini-UD-Q4_K_XL-Abliterated.gguf",
  [int]$MaxContext = 32768,
  [int]$Chunk = 512,
  [int]$Port = 8085,
  [string]$TokenizerDir = "C:\models\inkling\tokenizer\tokenizer",
  [string]$TokensFile = "C:\models\inkling\pw0.txt",
  [int]$MaxNew = 128,
  [switch]$Lan,
  [string]$ApiKey = "",
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
$bindAddr = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
$cfg = [ordered]@{
  arch           = "inkling2"
  model_name     = [IO.Path]::GetFileNameWithoutExtension($Model)
  exe            = $exe
  args           = @("--model",$Model,"--kv-q8","--max-context","$MaxContext","--chunk","$Chunk","--serve")
  tokenizer      = $TokenizerDir
  env            = $env_cfg
  fit_max_tokens = $true
}
if ($ApiKey) { $env:STRATA_API_KEY = $ApiKey } else { $cfg.api_key = "none" }
$cfgPath = Join-Path $root "inkling-run.json"
$cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $cfgPath
"Inkling: contesto $MaxContext, chunk $Chunk, ascolto su ${bindAddr}:$Port (attendi ~40 s)"
if ($Lan) {
  Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' } |
    ForEach-Object { "  API per Open WebUI: http://$($_.IPAddress):$Port/v1   ($($_.InterfaceAlias))" }
}
& (Join-Path $root "terminal\bin\strata.cmd") serve --config $cfgPath --host $bindAddr --port $Port
