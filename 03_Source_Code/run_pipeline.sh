#!/usr/bin/env bash
# One-shot setup on Linux/macOS:  ./run_pipeline.sh [rows]
set -euo pipefail
ROWS=${1:-100000}
python -m pip install -r requirements.txt
python -m data.ingest
python -m data.build_index --rows "$ROWS"
python -m data.train_models
python -m pytest
echo "Done. Start API: uvicorn app.main:app --port 8000 ; UI: streamlit run frontend/streamlit_app.py"
