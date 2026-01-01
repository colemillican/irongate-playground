#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT="${IRON_GATE_PERSIST_ROOT:-/workspace/irongate}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] Repo: $REPO_DIR"
echo "[bootstrap] Persist root: $PERSIST_ROOT"

# OS deps
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl git ca-certificates python3-venv python3-pip unzip

# Persistence dirs
mkdir -p "$PERSIST_ROOT"/{logs,pids,qdrant,ollama_models}

# Python venv + deps
cd "$REPO_DIR"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ollama install (if missing)
if ! command -v ollama >/dev/null 2>&1; then
  echo "[bootstrap] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "[bootstrap] Ollama already installed"
fi

# Pull model(s) (optional but recommended)
# Only works once ollama serve is running, so we’ll start ollama temporarily if needed.
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "[bootstrap] Starting ollama temporarily to pull models..."
  OLLAMA_MODELS="$PERSIST_ROOT/ollama_models" nohup ollama serve > "$PERSIST_ROOT/logs/ollama.log" 2>&1 &
  sleep 2
fi

echo "[bootstrap] Pulling model: qwen2.5:14b"
OLLAMA_MODELS="$PERSIST_ROOT/ollama_models" ollama pull qwen2.5:14b || true

echo "[bootstrap] Pulling embed model: nomic-embed-text"
OLLAMA_MODELS="$PERSIST_ROOT/ollama_models" ollama pull nomic-embed-text || true

# Qdrant install (binary)
mkdir -p "$PERSIST_ROOT/qdrant"
cd "$PERSIST_ROOT/qdrant"
if [ ! -x "$PERSIST_ROOT/qdrant/qdrant" ]; then
  echo "[bootstrap] Downloading Qdrant..."
  curl -L -o qdrant.tar.gz https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf qdrant.tar.gz
  chmod +x qdrant
else
  echo "[bootstrap] Qdrant already installed"
fi

# NOTE: Qdrant stores data relative to where it runs by default.
# We will run it from $PERSIST_ROOT/qdrant so storage persists.

echo
echo "[bootstrap] Done."
echo "[bootstrap] Start everything with:"
echo "  IRON_GATE_PERSIST_ROOT=$PERSIST_ROOT bash $REPO_DIR/scripts/run_all.sh"
