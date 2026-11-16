param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$SkipAdbReverse
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

if (-not $SkipAdbReverse) {
    $AdbPath = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
    if (Test-Path $AdbPath) {
        try {
            & $AdbPath reverse "tcp:$Port" "tcp:$Port" | Out-Null
            Write-Host "[ADB] USB reverse tunnel active on tcp:$Port"
        } catch {
            Write-Host "[ADB] No USB Android device detected; continuing with LAN/localhost mode."
        }
    }
}

Write-Host "[SERVER] Starting Stock-to-Stove FastAPI Server on http://${BindHost}:${Port}"
python -m uvicorn src.server.app:app --host $BindHost --port $Port
