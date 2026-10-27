param([string]$Device = "cpu", [switch]$SkipLpips)
$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $PythonExe = ".venv/Scripts/python.exe"
    function Invoke-Sigk {
        & $PythonExe -X utf8 -m sigk @args
        if ($LASTEXITCODE -ne 0) { throw "Komenda SIGK zakończona błędem: $args" }
    }
    Invoke-Sigk demo-data --data data/demo
    foreach ($Condition in @("sr4", "sr8", "dn001", "dn003")) {
        Invoke-Sigk train --condition $Condition --data data/demo --runs runs/smoke --epochs 2 --repeat 1 --width 8 --blocks 1 --batch-size 2 --device $Device
    }
    $EvalArgs = @("evaluate", "--data", "data/demo", "--runs", "runs/smoke", "--output", "runs/smoke-evaluation", "--examples", 2, "--device", $Device)
    if ($SkipLpips) { $EvalArgs += "--skip-lpips" }
    Invoke-Sigk @EvalArgs
} finally { Pop-Location }
