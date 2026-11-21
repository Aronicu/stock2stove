param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$BuildFrontend,
    [switch]$SkipFrontend,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
$env:FORWARDED_ALLOW_IPS = "*"

$PythonBin = "python"
$VenvWinPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$VenvUnixPython = Join-Path $ProjectRoot "venv\bin\python"
if (Test-Path $VenvWinPython) {
    $PythonBin = $VenvWinPython
} elseif (Test-Path $VenvUnixPython) {
    $PythonBin = $VenvUnixPython
}

$FrontendDir = Join-Path $ProjectRoot "frontend"
$FrontendPkg = Join-Path $FrontendDir "package.json"
$FrontendNodeModules = Join-Path $FrontendDir "node_modules"
$FrontendDistIndex = Join-Path $FrontendDir "dist\index.html"

if ((-not $SkipFrontend) -and (Test-Path $FrontendPkg)) {
    $NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $NpmCmd) {
        if ($Dev -or (-not (Test-Path $FrontendDistIndex))) {
            throw "npm is required to build or run the frontend."
        }
        Write-Host "[FRONTEND] npm not found; using existing frontend/dist/index.html"
    } else {
        if (-not (Test-Path $FrontendNodeModules)) {
            Write-Host "[FRONTEND] Installing npm dependencies in frontend/"
            Push-Location $FrontendDir
            try {
                & $NpmCmd.Source install
                if ($LASTEXITCODE -ne 0) {
                    throw "npm install failed with exit code $LASTEXITCODE"
                }
            } finally {
                Pop-Location
            }
        }

        if (-not $Dev) {
            $NeedBuild = $false
            if ($BuildFrontend -or (-not (Test-Path $FrontendDistIndex))) {
                $NeedBuild = $true
            } else {
                $DistTime = (Get-Item $FrontendDistIndex).LastWriteTime
                $SrcDir = Join-Path $FrontendDir "src"
                if (Test-Path $SrcDir) {
                    $NewerFiles = Get-ChildItem -Path $SrcDir -Recurse -File | Where-Object { $_.LastWriteTime -gt $DistTime }
                    if ($NewerFiles) {
                        $NeedBuild = $true
                    }
                }
                foreach ($WatchFile in @("index.html", "package.json", "vite.config.js")) {
                    $WatchPath = Join-Path $FrontendDir $WatchFile
                    if ((Test-Path $WatchPath) -and ((Get-Item $WatchPath).LastWriteTime -gt $DistTime)) {
                        $NeedBuild = $true
                    }
                }
            }

            if ($NeedBuild) {
                Write-Host "[FRONTEND] Building React SPA in frontend/dist"
                Push-Location $FrontendDir
                try {
                    & $NpmCmd.Source run build
                    if ($LASTEXITCODE -ne 0) {
                        throw "npm run build failed with exit code $LASTEXITCODE"
                    }
                } finally {
                    Pop-Location
                }
            }
        }
    }
}

Write-Host "[SERVER] Local UI: http://localhost:${Port}"
Write-Host "[SERVER] Starting Stock-to-Stove FastAPI Server on http://${BindHost}:${Port}"

if ($Dev -and (-not $SkipFrontend) -and (Test-Path $FrontendPkg)) {
    $NpmExec = (Get-Command npm -ErrorAction Stop).Source
    $BackendProc = $null
    $FrontendProc = $null
    try {
        $BackendProc = Start-Process -FilePath $PythonBin -ArgumentList @("-m", "uvicorn", "src.server.app:app", "--host", $BindHost, "--port", "$Port", "--proxy-headers") -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru
        Write-Host "[DEV] Starting Vite SPA dev server in frontend/ (http://localhost:5173)"
        $FrontendProc = Start-Process -FilePath $NpmExec -ArgumentList @("run", "dev") -WorkingDirectory $FrontendDir -NoNewWindow -PassThru
        while ((-not $BackendProc.HasExited) -and (-not $FrontendProc.HasExited)) {
            Start-Sleep -Milliseconds 500
        }
    } finally {
        if ($FrontendProc -and (-not $FrontendProc.HasExited)) {
            Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
        }
        if ($BackendProc -and (-not $BackendProc.HasExited)) {
            Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    & $PythonBin -m uvicorn src.server.app:app --host $BindHost --port $Port --proxy-headers
}
