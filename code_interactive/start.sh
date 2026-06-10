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
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
#   cp .env.example .env   # then fill OPENAI_API_KEY
#
# 모델 / 대화 설정:
#   Agent behavior is configured in agents/agent_config.py.
#   Web app and OpenAI runtime settings live in web_app_config.py.
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
echo "  LLM   : ChatGPT (gpt-5.2)"
if [ -n "$RELOAD_FLAG" ]; then
echo "  Mode  : DEV (hot-reload ON)"
else
echo "  Mode  : Production"
fi
echo "==================================================="

# 이 스크립트가 있는 디렉터리(code_interactive/)로 이동
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}

# shellcheck disable=SC2086
"$PYTHON" -m uvicorn app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    $RELOAD_FLAG
