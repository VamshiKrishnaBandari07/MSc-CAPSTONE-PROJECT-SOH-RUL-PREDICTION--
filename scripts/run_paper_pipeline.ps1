# Paper experiment — NASA + Oxford + CALCE (5-fold CV)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Paper reproduction (3 datasets) ===" -ForegroundColor Cyan
git lfs pull 2>$null

python run_paper_experiment.py --require-real --cpu
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python generate_figures.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/verify_repo.py
python -m pytest tests/ -v --tb=short
Write-Host "=== Complete ===" -ForegroundColor Green
