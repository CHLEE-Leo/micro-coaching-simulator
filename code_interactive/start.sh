#!/usr/bin/env bash
# ================================================================
# start.sh - Launch the Micro-Coaching Interactive Web Server
# ================================================================
# Usage:
#   ./start.sh              # production mode, default port 8000
#   ./start.sh 8080         # production mode, custom port
#   DEV=1 ./start.sh        # development mode (--reload hot-reload)
#   DEV=1 ./start.sh 8080   # dev mode on custom port
#
# Requirements:
#   conda activate micro-coaching-chatbot
#
# 모델 / 대화 설정:
#   code/config.py 의 SimulationConfig 를 수정하세요.
#   (max_turns, context_window 등 대화 제어 파라미터 포함)
#   gguf_path 는 config_interactive.py 에서 .gguf 파일 경로로 지정합니다.
# ================================================================

set -e

PORT=${1:-8000}

RELOAD_FLAG=""
if [ "${DEV:-0}" = "1" ]; then
    RELOAD_FLAG="--reload"
fi

echo "==================================================="
echo "  Micro-Coaching Simulator  |  Interactive Mode"
echo "---------------------------------------------------"
echo "  Port  : $PORT"
echo "  URL   : http://localhost:$PORT"
echo "  LLM   : llama-cpp-python (GPU if n_gpu_layers=-1)"
if [ -n "$RELOAD_FLAG" ]; then
echo "  Mode  : DEV (hot-reload ON)"
else
echo "  Mode  : Production"
fi
echo "==================================================="

# 이 스크립트가 있는 디렉터리(code_interactive/)로 이동
cd "$(dirname "$0")"

PYTHON=/home/messy92/anaconda3/envs/micro-coaching-chatbot/bin/python

# shellcheck disable=SC2086
"$PYTHON" -m uvicorn app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    $RELOAD_FLAG
