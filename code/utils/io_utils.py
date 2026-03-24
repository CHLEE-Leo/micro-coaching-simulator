"""
utils/io_utils.py
─────────────────
데이터 입출력 공통 유틸리티.
  - 데이터셋 로딩 (new_meals.csv)
  - 생성 결과 저장 / 이어쓰기
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터셋 로딩
# ──────────────────────────────────────────────────────────────────────────────

def load_meal_data(data_path: str, target_goal: str) -> pd.DataFrame:
    """
    df_normal_without_test_string.csv 를 로드한 뒤 목표별·전문가 검증 통과 샘플만 반환합니다.

    원본 컬럼 → 내부 표준 컬럼 매핑:
      short_name  → goal_type
      meal_id     → id
      kind        → meal_type
      title_y     → meal_description

    Parameters
    ----------
    data_path    : CSV 파일 경로
    target_goal  : "drink_water" | "lean_protein" | "half_fruits_vegetables" | "one_fourth_carbs"

    Returns
    -------
    pd.DataFrame : [goal_id, goal_type, id, meal_type, meal_description, meal_ingredient, expert_result]
    """
    df = pd.read_csv(data_path)

    # 컬럼명 표준화
    df = df.rename(columns={
        "short_name":    "goal_type",
        "meal_id":       "id",
        "kind":          "meal_type",
        "title_y":       "meal_description",
        "ingredients_y": "meal_ingredient",
    })

    # ingredients_y 가 없는 행은 빈 문자열로 처리
    if "meal_ingredient" in df.columns:
        df["meal_ingredient"] = df["meal_ingredient"].fillna("")
    else:
        df["meal_ingredient"] = ""

    # 스낵 시간대를 통일된 레이블로 정규화
    df["meal_type"] = df["meal_type"].replace({
        "after_dinner_snack": "a snack",
        "afternoon_snack":    "a snack",
        "morning_snack":      "a snack",
    })

    target_cols = ["goal_id", "goal_type", "id", "meal_type", "meal_description", "meal_ingredient", "expert_result"]
    filtered = df[df["goal_type"] == target_goal][target_cols].copy().reset_index(drop=True)

    print(f"[io_utils] '{target_goal}' 목표의 전체 샘플 수: {len(filtered)}")
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# 2. 결과 저장
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_results(output_path: str) -> List[Any]:
    """
    기존 결과 파일이 있으면 로드하고 없으면 빈 리스트를 반환합니다.
    이어쓰기(append) 모드를 지원합니다.

    Parameters
    ----------
    output_path : JSON 파일 경로

    Returns
    -------
    list : 기존에 저장된 결과 리스트
    """
    if not os.path.exists(output_path):
        print(f"[io_utils] 새로 시작합니다: {output_path}")
        return []

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[io_utils] 기존 결과 {len(data)}건 로드: {output_path}")
            return data
    except (json.JSONDecodeError, ValueError):
        print(f"[io_utils] 파일 파싱 실패, 새로 시작합니다: {output_path}")
        return []


def save_results(data: List[Any], output_path: str) -> None:
    """
    결과 리스트를 JSON 파일로 저장합니다.
    상위 디렉토리가 없으면 자동 생성합니다.

    Parameters
    ----------
    data        : 저장할 리스트
    output_path : 저장 경로
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_output_path(
    base_dir: str,
    goal: str,
    model_name: str,
    max_turns: int,
    suffix: str = "",
) -> str:
    """
    표준 출력 경로를 생성합니다.

    구조:
      {base_dir}/goal={goal}/model={model_name}/llm_coach_dialogs(...).json

    Parameters
    ----------
    base_dir   : 루트 출력 디렉토리 (예: "../results")
    goal       : 영양 목표
    model_name : 모델 이름 (HuggingFace repo의 마지막 부분)
    max_turns  : 최대 턴 수
    suffix     : 파일명 접미사 (예: "_no_guidance")

    Returns
    -------
    str : 완성된 파일 경로
    """
    filename = f"llm_coach_dialogs(goal={goal}_{max_turns}_turns){suffix}.json"
    return os.path.join(
        base_dir,
        f"goal={goal}",
        f"model={model_name}",
        filename,
    )
