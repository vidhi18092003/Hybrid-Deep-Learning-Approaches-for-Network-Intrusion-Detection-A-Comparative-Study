# Windows PowerShell Demo Runner for Streaming Hybrid IDS
# This script executes tests, prepares data, trains models, evaluates, and starts streaming.

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Starting Hybrid Anomaly Detection Pipeline Demo" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Setup Directories
Write-Host "[1/6] Creating necessary data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "data/stream_input" | Out-Null
New-Item -ItemType Directory -Force -Path "data/checkpoint" | Out-Null

# 2. Run Tests
Write-Host "[2/6] Running unit tests..." -ForegroundColor Yellow
python -m pytest tests/ -q --no-header
if ($LASTEXITCODE -ne 0) {
    Write-Host "Unit tests failed! Aborting demo execution." -ForegroundColor Red
    Exit $LASTEXITCODE
}
Write-Host "Unit tests passed successfully!" -ForegroundColor Green

# 3. Preprocess Datasets
if (-not (Test-Path "data/cicids2017_train.parquet")) {
    Write-Host "[3/6] Running Data Engineering and Unified Feature Schema Layer..." -ForegroundColor Yellow
    python src/ingestion/preprocess.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Preprocessing failed! Aborting demo." -ForegroundColor Red
        Exit $LASTEXITCODE
    }
} else {
    Write-Host "[3/6] Preprocessed dataset already exists in data/. Skipping preprocessing." -ForegroundColor Green
}

# 4. Train Models
if (-not (Test-Path "data/bilstm_model.pt")) {
    Write-Host "[4/6] Training Models (MLP Baseline, 1D-CNN, BiLSTM, and Hybrid GRU)..." -ForegroundColor Yellow
    python src/models/train_initial.py
    python src/models/train_all_models.py
} else {
    Write-Host "[4/6] All deep learning models already trained in data/. Skipping training." -ForegroundColor Green
}

# 5. Run Evaluation Harness
if (-not (Test-Path "data/comparison_table_cicids2017.csv")) {
    Write-Host "[5/6] Benchmarking all 3 DL models on CICIDS2017 and UNSW-NB15..." -ForegroundColor Yellow
    python src/evaluation/evaluator.py
    python src/evaluation/compare_models.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Model evaluation failed! Aborting demo." -ForegroundColor Red
        Exit $LASTEXITCODE
    }
} else {
    Write-Host "[5/6] Comparison benchmarks already generated in data/. Skipping evaluation." -ForegroundColor Green
}

# 6. Launch Streaming Pipeline
Write-Host "[6/6] Launching Streaming Pipeline and Dashboard..." -ForegroundColor Yellow

# Clean any existing live results
if (Test-Path "data/live_results.json") {
    Remove-Item "data/live_results.json" -Force
}

# Ensure background logs directory exists
New-Item -ItemType Directory -Force -Path "data/logs" | Out-Null

Write-Host "Starting Streaming Consumer in background..." -ForegroundColor Yellow
$ConsumerProc = Start-Process python -ArgumentList "src/streaming/consumer.py" -RedirectStandardOutput "data/logs/consumer.log" -RedirectStandardError "data/logs/consumer_err.log" -WindowStyle Hidden -PassThru

# Give consumer a moment to boot
Start-Sleep -Seconds 2

Write-Host "Starting Simulated Traffic Producer in background..." -ForegroundColor Yellow
$ProducerProc = Start-Process python -ArgumentList "src/ingestion/producer.py --dataset cicids2017_test.parquet --rate 50 --loop" -RedirectStandardOutput "data/logs/producer.log" -RedirectStandardError "data/logs/producer_err.log" -WindowStyle Hidden -PassThru

Write-Host "Starting Streamlit Dashboard in foreground..." -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate dashboard and background tasks when done." -ForegroundColor Cyan

# Start Streamlit in foreground
streamlit run dashboard/app.py --server.headless true

# Cleanup on exit
Write-Host "Stopping background processes..." -ForegroundColor Yellow
if ($ProducerProc -and -not $ProducerProc.HasExited) {
    Stop-Process -Id $ProducerProc.Id -Force -ErrorAction SilentlyContinue
}
if ($ConsumerProc -and -not $ConsumerProc.HasExited) {
    Stop-Process -Id $ConsumerProc.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "Demo shutdown complete." -ForegroundColor Green
