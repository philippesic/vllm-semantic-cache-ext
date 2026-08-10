param(
    [ValidateSet("smoke", "full")]
    [string]$RunKind = "smoke",
    [string]$SessionRoot = "C:\Users\philip\semantic-cache-agent\session-20260809"
)

$ErrorActionPreference = "Stop"
$maxCases = if ($RunKind -eq "smoke") { 1 } else { 4 }
$script = "$SessionRoot\inputs\experiments\step_1_4_blend_sweep.py"
$output = "$SessionRoot\outputs\step_1_4_blend_sweep_${RunKind}.csv"
$runner = "$SessionRoot\tools\run_python.ps1"
$arguments = @(
    "--max-cases",
    "$maxCases",
    "--results-csv",
    $output
)

& $runner `
    -ScriptPath $script `
    -RunName "cp014_blend_sweep_$RunKind" `
    -ScriptArgs $arguments `
    -SessionRoot $SessionRoot
