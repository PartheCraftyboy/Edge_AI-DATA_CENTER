#!/usr/bin/env bash
# Start Palimpsest: API and frontend on one port. Then open http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/backend"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
