$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
if (-not $ProjectRoot) {
    $ProjectRoot = (Get-Location).Path
}
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

$PythonBin = "python"
$VenvWinPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$VenvUnixPython = Join-Path $ProjectRoot "venv\bin\python"
if (Test-Path $VenvWinPython) {
    $PythonBin = $VenvWinPython
} elseif (Test-Path $VenvUnixPython) {
    $PythonBin = $VenvUnixPython
}

Write-Host "=========================================================="
Write-Host " STOCK-TO-STOVE E2E TEST RUNNER"
Write-Host " PROJECT_ROOT: $ProjectRoot"
Write-Host " PYTHON_BIN:   $PythonBin"
Write-Host "=========================================================="

Write-Host "[1/2] Running backend pytest suite..."
& $PythonBin -m pytest -v --basetemp=.pytest_tmp
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine("Backend pytest suite failed with exit code $LASTEXITCODE")
    exit $LASTEXITCODE
}

$FrontendDir = Join-Path $ProjectRoot "frontend"
$FrontendPkg = Join-Path $FrontendDir "package.json"
$FrontendNodeModules = Join-Path $FrontendDir "node_modules"
$FrontendDistDir = Join-Path $FrontendDir "dist"

if (-not (Test-Path $FrontendPkg)) {
    [Console]::Error.WriteLine("Missing frontend/package.json at $FrontendPkg")
    exit 1
}

$NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $NpmCmd) {
    [Console]::Error.WriteLine("npm is required to verify the frontend build.")
    exit 1
}

if (-not (Test-Path $FrontendNodeModules)) {
    Write-Host "[FRONTEND] Installing npm dependencies in frontend/..."
    Push-Location $FrontendDir
    try {
        & $NpmCmd.Source install
        if ($LASTEXITCODE -ne 0) {
            [Console]::Error.WriteLine("npm install failed with exit code $LASTEXITCODE")
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

Write-Host "[2/2] Running frontend Vite production build..."
Push-Location $FrontendDir
try {
    & $NpmCmd.Source run build
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("npm run build failed with exit code $LASTEXITCODE")
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

$ExpectedDistFiles = @(
    "index.html",
    "manifest.webmanifest",
    "icons\icon-192.png",
    "icons\icon-512.png"
)
foreach ($RelPath in $ExpectedDistFiles) {
    $FullPath = Join-Path $FrontendDistDir $RelPath
    if (-not (Test-Path $FullPath)) {
        [Console]::Error.WriteLine("Expected built frontend artifact not found: $FullPath")
        exit 1
    }
}

$AssetsDir = Join-Path $FrontendDistDir "assets"
if (-not (Test-Path $AssetsDir)) {
    [Console]::Error.WriteLine("Expected built frontend assets directory not found: $AssetsDir")
    exit 1
}

Write-Host "E2E verification completed successfully (backend pytest + frontend build)."
exit 0
