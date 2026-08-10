param(
    [string]$SessionRoot = "C:\Users\philip\semantic-cache-agent\session-20260809"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = "$SessionRoot\logs\install_torch_$timestamp.log"

Start-Transcript -Path $transcript
try {
    $env:UV_CACHE_DIR = "$SessionRoot\uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = "$SessionRoot\python"
    $env:UV_TOOL_DIR = "$SessionRoot\uv-tools"
    $env:UV_TOOL_BIN_DIR = "$SessionRoot\tools\bin"
    $env:UV_NO_MODIFY_PATH = "1"

    $uv = "$SessionRoot\tools\uv\uv.exe"
    $python = "$SessionRoot\workspace\.venv\Scripts\python.exe"
    & $uv pip install --python $python torch==2.12.0 `
        --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch install failed with exit code $LASTEXITCODE"
    }
    & $python "$PSScriptRoot\verify_torch.py"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript
}
