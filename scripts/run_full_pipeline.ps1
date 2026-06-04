# MSc Capstone — full experiment pipeline (Phase 1 -> Phase 2 -> figures)
Set-Location $PSScriptRoot\..

$log = "results\full_pipeline.log"
New-Item -ItemType Directory -Force -Path results | Out-Null
[System.IO.File]::WriteAllText((Join-Path (Get-Location) $log), "", (New-Object System.Text.UTF8Encoding $false))

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

$env:PYTHONUNBUFFERED = "1"

Log "=== PHASE 1: Paper experiment (5-fold CV) ==="
python run_paper_experiment.py --require-real --cpu *>> $log
if ($LASTEXITCODE -ne 0) { Log "PHASE 1 FAILED exit $LASTEXITCODE"; exit $LASTEXITCODE }

Log "=== PHASE 2: MSc extension (B + C) ==="
python run_experiments.py --msc-only --require-real --cpu *>> $log
if ($LASTEXITCODE -ne 0) { Log "PHASE 2 FAILED exit $LASTEXITCODE"; exit $LASTEXITCODE }

Log "=== FIGURES ==="
python generate_figures.py *>> $log
if ($LASTEXITCODE -ne 0) { Log "FIGURES FAILED exit $LASTEXITCODE"; exit $LASTEXITCODE }

Log "=== SYNC DOCS ==="
python scripts/sync_results_docs.py *>> $log
if ($LASTEXITCODE -ne 0) { Log "SYNC DOCS FAILED exit $LASTEXITCODE"; exit $LASTEXITCODE }

Log "=== PIPELINE COMPLETE ==="
exit 0
