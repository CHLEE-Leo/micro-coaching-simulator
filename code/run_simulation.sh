#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_simulation.sh
# LLM Coach ↔ User ↔ Judge 영양 코칭 대화 시뮬레이션 실행 스크립트
#
# 모든 실험 설정(goal, model, turns 등)은 config.py 에서 직접 수정하세요.
# 이 스크립트는 환경 변수(GPU, Conda 환경)만 담당합니다.
#
# 사용법:
#   bash run_simulation.sh [gpu]
#   예) bash run_simulation.sh 7        # GPU 7번만 사용
#       bash run_simulation.sh 6,7      # GPU 6,7번 사용
#       bash run_simulation.sh           # GPU 기본값 (config.py num_gpus 참조)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 환경 설정 ──────────────────────────────────────────────────────────────────
GPU="${1:-0}"          # 1번 인자: CUDA_VISIBLE_DEVICES (예: 7 또는 6,7)
CONDA_ENV="micro-coaching-chatbot"

# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo "  GPU(s)      : ${GPU}"
echo "  Conda env   : ${CONDA_ENV}"
echo "  config.py 에서 실험 설정을 확인하세요."
echo "======================================================"
echo ""

cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
# vLLM(libgomp) 과 Intel MKL 스레딩 레이어 충돌 방지
export MKL_THREADING_LAYER=GNU

echo "[*] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[*] MKL_THREADING_LAYER=${MKL_THREADING_LAYER}"
echo ""

conda run -n "${CONDA_ENV}" --no-capture-output \
    python run_simulation.py

echo ""
echo "[Done]"
