param(
    [ValidateSet('all','check','download','prepare','train','evaluate','report')]
    [string]$Stage = 'all',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Options
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $candidates = @('python', 'py', (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'))
    $found = $false
    foreach ($candidate in $candidates) {
        try {
            & $candidate -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,14) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) {
                & $candidate -m venv .venv
                if ($LASTEXITCODE -ne 0) { throw 'Nie udalo sie utworzyc srodowiska.' }
                $found = $true
                break
            }
        } catch { continue }
    }
    if (-not $found) { throw 'Zainstaluj Python 3.11, 3.12 lub 3.13 i uruchom ponownie.' }
}
& $pythonPath -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Nie udalo sie przygotowac pip.' }
}
$hash = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'requirements.txt') -Algorithm SHA256).Hash
$stamp = Join-Path $PSScriptRoot '.venv\requirements.sha256'
if (-not (Test-Path -LiteralPath $stamp) -or (Get-Content -LiteralPath $stamp -Raw).Trim() -ne $hash) {
    & $pythonPath -m pip install --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Instalacja zaleznosci nie powiodla sie.' }
    Set-Content -LiteralPath $stamp -Value $hash
}
if ($Stage -eq 'report') { & $pythonPath report.py @Options }
else { & $pythonPath main.py $Stage @Options }
if ($LASTEXITCODE -ne 0) { throw 'Program zakonczyl sie bledem. Popraw przyczyne i uruchom ponownie.' }
