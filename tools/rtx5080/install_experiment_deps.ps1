param(
    [string]$SessionRoot = "C:\Users\philip\semantic-cache-agent\session-20260809"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = "$SessionRoot\logs\install_experiment_deps_$timestamp.log"

Start-Transcript -Path $transcript
try {
    $env:UV_CACHE_DIR = "$SessionRoot\uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = "$SessionRoot\python"
    $env:UV_TOOL_DIR = "$SessionRoot\uv-tools"
    $env:UV_TOOL_BIN_DIR = "$SessionRoot\tools\bin"
    $env:UV_NO_MODIFY_PATH = "1"
    $env:HF_HOME = "$SessionRoot\artifacts\huggingface"

    $uv = "$SessionRoot\tools\uv\uv.exe"
    $python = "$SessionRoot\workspace\.venv\Scripts\python.exe"
    & $uv pip install --python $python numpy==2.5.1 transformers==5.14.1
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment dependency install failed with exit code $LASTEXITCODE"
    }
    & $uv pip freeze --python $python |
        Out-File -Encoding utf8 "$SessionRoot\outputs\requirements_$timestamp.txt"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency manifest failed with exit code $LASTEXITCODE"
    }
    & $python -c "import numpy, transformers; print(numpy.__version__, transformers.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency import check failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript
}
