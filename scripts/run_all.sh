#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT="${IRON_GATE_PERSIST_ROOT:-/workspace/irongate}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PERSIST_ROOT"/{logs,pids,qdrant}
mkdir -p "$REPO_DIR"/.venv

echo "[run_all] Repo: $REPO_DIR"
echo "[run_all] Persist: $PERSIST_ROOT"

# 1) Ollama
if ! command -v ollama >/dev/null 2>&1; then
  echo "[run_all] ERROR: ollama not installed. Run scripts/bootstrap.sh"
  exit 1
fi

if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "[run_all] starting ollama..."
  OLLAMA_MODELS="$PERSIST_ROOT/ollama_models" nohup ollama serve > "$PERSIST_ROOT/logs/ollama.log" 2>&1 &
  echo $! > "$PERSIST_ROOT/pids/ollama.pid"
else
  echo "[run_all] ollama already running"
fi

# 2) Qdrant
QDRANT_BIN="$PERSIST_ROOT/qdrant/qdrant"
if [ ! -x "$QDRANT_BIN" ]; then
  echo "[run_all] ERROR: qdrant binary missing at $QDRANT_BIN. Run scripts/bootstrap.sh"
  exit 1
fi

if ! pgrep -f "$QDRANT_BIN" >/dev/null 2>&1; then
  echo "[run_all] starting ollama..."
  OLLAMA_MODELS="$PERSIST_ROOT/ollama_models" nohup ollama serve > "$PERSIST_ROOT/logs/ollama.log" 2>&1 &
  echo $! > "$PERSIST_ROOT/pids/ollama.pid"
else
  echo "[run_all] ollama already running"
fi

# 2) Qdrant
QDRANT_BIN="$PERSIST_ROOT/qdrant/qdrant"
if [ ! -x "$QDRANT_BIN" ]; then
  echo "[run_all] ERROR: qdrant binary missing at $QDRANT_BIN. Run scripts/bootstrap.sh"
  exit 1
fi

if ! pgrep -f "$QDRANT_BIN" >/dev/null 2>&1; then
  echo "[run_all] starting qdrant..."
  mkdir -p "$PERSIST_ROOT/qdrant/storage"
  nohup "$QDRANT_BIN" > "$PERSIST_ROOT/logs/qdrant.log" 2>&1 &
  echo $! > "$PERSIST_ROOT/pids/qdrant.pid"
else
  echo "[run_all] qdrant already running"
fi

# 3) API
if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
  echo "[run_all] ERROR: .venv missing. Run scripts/bootstrap.sh"
  exit 1
fi

if ! pgrep -f "uvicorn api.main:app" >/dev/null 2>&1; then
  echo "[run_all] starting api..."
  cd "$REPO_DIR"
  nohup "$REPO_DIR/.venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 > "$PERSIST_ROOT/logs/uvicorn.log" 2>&1 &
  echo $! > "$PERSIST_ROOT/pids/api.pid"
else
  echo "[run_all] api already running"
fi

echo
echo "[run_all] HEALTH CHECKS"
curl -sS http://127.0.0.1:8000/health && echo
curl -sS http://127.0.0.1:6333/readyz && echo
