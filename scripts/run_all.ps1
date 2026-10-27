param(
    [int]$Epochs = 60,
    [int]$Width = 32,
    [int]$Blocks = 6,
    [int]$Repeat = 4,
    [int]$BatchSize = 8,
    [ValidateSet("fp32", "bf16")][string]$DenoisePrecision = "fp32",
    [string]$Data = "data/div2k256",
    [string]$Runs = "runs/main",
    [string]$Output = "runs/evaluation",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [switch]$Download,
    [switch]$Resume,
    [string]$SneDefinition = ""
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $ProjectRoot
try {
    $PythonExe = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Brak .venv. Zobacz README.md." }
    function Invoke-Sigk {
        & $PythonExe -X utf8 -m sigk @args
        if ($LASTEXITCODE -ne 0) { throw "Komenda SIGK zakończona błędem: $args" }
    }
    Invoke-Sigk doctor
    if ($Download) { Invoke-Sigk download }
    if (-not (Test-Path -LiteralPath (Join-Path $Data "manifest.json"))) {
        Invoke-Sigk prepare --data $Data
    }
    Invoke-Sigk verify --data $Data
    foreach ($Condition in @("sr4", "sr8", "dn001", "dn003")) {
        $Precision = if ($Condition.StartsWith("dn")) { $DenoisePrecision } else { "fp32" }
        $TrainArgs = @("train", "--condition", $Condition, "--data", $Data, "--runs", $Runs,
                       "--epochs", $Epochs, "--device", $Device, "--width", $Width,
                       "--blocks", $Blocks, "--repeat", $Repeat, "--batch-size", $BatchSize,
                       "--precision", $Precision)
        if ($Resume -and (Test-Path -LiteralPath (Join-Path $Runs "$Condition/last.pt"))) {
            $TrainArgs += "--resume"
        }
        Invoke-Sigk @TrainArgs
    }
    $BilateralConfig = Join-Path $Runs "bilateral.json"
    Invoke-Sigk tune-baseline --data $Data --output $BilateralConfig
    $EvalArgs = @("evaluate", "--data", $Data, "--runs", $Runs, "--output", $Output,
                  "--bilateral-config", $BilateralConfig, "--device", $Device)
    if ($SneDefinition) { $EvalArgs += @("--sne-definition", $SneDefinition) }
    Invoke-Sigk @EvalArgs
} finally { Pop-Location }
