# Prefer a user-selected Python, then PATH, then the locally available runtime.
$ErrorActionPreference = 'Stop'
$computePython = $env:SUPERPOD_PYTHON
if (-not $computePython) {
    $foundPython = Get-Command python -ErrorAction SilentlyContinue
    if ($foundPython) { $computePython = $foundPython.Source }
}
if (-not $computePython) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $computePython = $bundledPython }
}
if (-not $computePython) { throw 'Python 3.10+ required. Set SUPERPOD_PYTHON to its executable path.' }
Push-Location -LiteralPath $PSScriptRoot
try { & $computePython -m superpod_compute @args; $computeExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $computeExitCode
