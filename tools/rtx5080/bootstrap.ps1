param(
    [string]$SessionRoot = "C:\Users\philip\semantic-cache-agent\session-20260809"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$paths = @(
    $SessionRoot,
    "$SessionRoot\artifacts",
    "$SessionRoot\inputs",
    "$SessionRoot\logs",
    "$SessionRoot\outputs",
    "$SessionRoot\python",
    "$SessionRoot\tools",
    "$SessionRoot\uv-cache",
    "$SessionRoot\uv-tools",
    "$SessionRoot\workspace"
)
New-Item -ItemType Directory -Force -Path $paths | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = "$SessionRoot\logs\bootstrap_$timestamp.log"
Start-Transcript -Path $transcript
try {
    $env:UV_UNMANAGED_INSTALL = "$SessionRoot\tools\uv"
    $env:UV_CACHE_DIR = "$SessionRoot\uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = "$SessionRoot\python"
    $env:UV_TOOL_DIR = "$SessionRoot\uv-tools"
    $env:UV_TOOL_BIN_DIR = "$SessionRoot\tools\bin"
    $env:UV_NO_MODIFY_PATH = "1"

    Write-Output "session_root=$SessionRoot"
    Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    $installer = "$SessionRoot\artifacts\uv-install.ps1"
    Invoke-WebRequest -Uri "https://astral.sh/uv/0.11.32/install.ps1" -OutFile $installer
    Get-FileHash -Algorithm SHA256 $installer
    & powershell -NoProfile -ExecutionPolicy Bypass -File $installer
    if ($LASTEXITCODE -ne 0) {
        throw "uv installer failed with exit code $LASTEXITCODE"
    }

    $uv = "$SessionRoot\tools\uv\uv.exe"
    & $uv --version
    if ($LASTEXITCODE -ne 0) {
        throw "uv version check failed with exit code $LASTEXITCODE"
    }
    & $uv python install --no-bin 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "uv Python install failed with exit code $LASTEXITCODE"
    }
    & $uv venv --python 3.12 "$SessionRoot\workspace\.venv"
    if ($LASTEXITCODE -ne 0) {
        throw "venv creation failed with exit code $LASTEXITCODE"
    }
    & "$SessionRoot\workspace\.venv\Scripts\python.exe" --version
    if ($LASTEXITCODE -ne 0) {
        throw "venv Python check failed with exit code $LASTEXITCODE"
    }

    Get-ChildItem -Recurse -Depth 2 $SessionRoot |
        Select-Object FullName, Length, LastWriteTime
}
finally {
    Stop-Transcript
}
