param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string[]]$ScriptArgs = @(),
    [string]$SessionRoot = "C:\Users\philip\semantic-cache-agent\session-20260809"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = "$SessionRoot\logs\${RunName}_$timestamp.log"
$resolvedRoot = (Resolve-Path $SessionRoot).Path
$resolvedScript = (Resolve-Path $ScriptPath).Path
if (-not $resolvedScript.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ScriptPath must be inside SessionRoot"
}

Start-Transcript -Path $transcript
try {
    $env:HF_HOME = "$SessionRoot\artifacts\huggingface"
    $env:HF_HUB_CACHE = "$SessionRoot\artifacts\huggingface\hub"
    $env:TRANSFORMERS_CACHE = "$SessionRoot\artifacts\huggingface\transformers"
    $env:UV_CACHE_DIR = "$SessionRoot\uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = "$SessionRoot\python"
    $env:UV_NO_MODIFY_PATH = "1"

    $python = "$SessionRoot\workspace\.venv\Scripts\python.exe"
    Write-Output "run_name=$RunName"
    Write-Output "script=$resolvedScript"
    Write-Output "arguments=$($ScriptArgs -join ' ')"
    Get-FileHash -Algorithm SHA256 $resolvedScript
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version `
        --format=csv,noheader
    & $python $resolvedScript @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python run failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript
}
