# Paper reproduction pipeline (CPU-friendly)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Paper reproduction pipeline ===" -ForegroundColor Cyan
git lfs pull 2>$null

python run_paper_experiment.py --require-real --cpu
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python generate_figures.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/sync_results_docs.py
python -m pytest tests/ -v --tb=short
Write-Host "=== Done ===" -ForegroundColor Green
