# One-shot setup on Windows:  powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 [-Rows 100000]
param([int]$Rows = 100000)
$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
python -m data.ingest
python -m data.build_index --rows $Rows
python -m data.train_models
python -m pytest
Write-Host "Done. Start API:  uvicorn app.main:app --port 8000"
Write-Host "Start UI:         streamlit run frontend/streamlit_app.py"
