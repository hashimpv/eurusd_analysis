# Run from Task Scheduler: powershell.exe -ExecutionPolicy Bypass -File "...\scripts\run_daily_update.ps1"
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("daily_update_{0:yyyyMMdd}.log" -f (Get-Date))

& py -3 -m fetch.daily_update *>&1 | Tee-Object -FilePath $LogFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
