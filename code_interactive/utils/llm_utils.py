"""
code_interactive/utils/llm_utils.py
─────────────────────────────────────
Interactive 모드 전용 LLM 유틸리티.
/code/utils/llm_utils.py 와 완전히 동일한 함수 시그니처를 제공하지만
vLLM 대신 llama-cpp-python 을 사용합니다.

이 파일이 /code/utils/llm_utils.py 보다 sys.path 우선순위가 높으므로
CoachModel / JudgeModel 이 `from utils.llm_utils import generate_response` 를
실행할 때 이 파일의 함수를 가져갑니다.  /code 는 변경 없음.

/ Identical function signatures to /code/utils/llm_utils.py but powered by
  llama-cpp-python so it runs on the user's CPU (or GPU via n_gpu_layers).
  /code is completely untouched.
"""

from __future__ import annotations

from typing import List, Dict, Optional

from llama_cpp import Llama


# ─────────────────────────────────────────────────────────────────────────────
# 타입 별칭 / Type alias
# ─────────────────────────────────────────────────────────────────────────────
# vLLM 코드가 LLM 타입을 참조하는 경우를 대비해 별칭 노출
LLM = Llama


# ─────────────────────────────────────────────────────────────────────────────
# 1. 모델 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    gguf_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    # 아래는 /code 와의 시그니처 호환을 위해 무시하는 인자들
    tensor_parallel_size: int = 1,   # ignored (llama-cpp 는 단일 프로세스)
    max_model_len: int = 4096,       # → n_ctx 로 매핑
    dtype: str = "auto",             # ignored (GGUF 내 양자화 형식 사용)
) -> Llama:
    """
    GGUF 모델을 로드합니다.

    Parameters
    ----------
    gguf_path     : 로컬 .gguf 파일 경로 또는 HF repo 포맷 "repo/filename.gguf"
                    예) "~/.cache/models/gemma-3-12b-it-Q4_K_M.gguf"
    n_ctx         : 최대 컨텍스트 토큰 수 (기본 4096)
    n_gpu_layers  : GPU 에 올릴 레이어 수.
                    0 = CPU 전용, -1 = 모든 레이어를 GPU 에 올림.
                    CUDA/Metal 없이는 0 을 사용하세요.
    n_threads     : CPU 스레드 수. None 이면 자동 설정.
    verbose       : llama.cpp 디버그 로그 출력 여부

    Returns
    -------
    llama_cpp.Llama
    """
    # max_model_len 이 전달된 경우 n_ctx 우선
    effective_ctx = max(n_ctx, max_model_len)

    kwargs: dict = dict(
        model_path=gguf_path,
        n_ctx=effective_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        chat_format="chatml",   # 대부분 모델에 적합; gemma 는 chatml 호환
    )
    if n_threads is not None:
        kwargs["n_threads"] = n_threads

    return Llama(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SamplingParams 빌더 (내부용 — dict 반환)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sampling_kwargs(
    sampling: str = "greedy",
    max_new_tokens: int = 80,
) -> dict:
    """llama_cpp.create_chat_completion 에 전달할 키워드 인자 딕셔너리 반환."""
    if sampling in ("beam", "greedy"):
        return dict(
            temperature=0.0,
            max_tokens=max_new_tokens,
            repeat_penalty=1.15,
        )
    elif sampling == "sampling":
        return dict(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_new_tokens,
            repeat_penalty=1.15,
        )
    else:
        raise ValueError(
            f"지원하지 않는 sampling 방식: '{sampling}'. "
            "beam | greedy | sampling 중 선택하세요."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. 단일 응답 생성
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    llm: Llama,
    messages: List[Dict[str, str]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
) -> str:
    """
    단일 messages 리스트에 대해 응답을 생성합니다.
    /code/utils/llm_utils.py 의 generate_response 와 동일한 시그니처입니다.

    Parameters
    ----------
    llm            : llama_cpp.Llama 객체
    messages       : [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    sampling       : "beam" | "greedy" | "sampling"
    max_new_tokens : 최대 생성 토큰 수
    stop_at_newline: True 이면 첫 번째 비어 있지 않은 줄만 반환

    Returns
    -------
    str : 생성된 응답 텍스트
    """
    kwargs = _build_sampling_kwargs(sampling=sampling, max_new_tokens=max_new_tokens)

    response = llm.create_chat_completion(
        messages=messages,
        **kwargs,
    )
    raw: str = response["choices"][0]["message"]["content"] or ""

    if stop_at_newline:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[0] if lines else ""
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. 배치 응답 생성
#    interactive 모드는 배치 추론이 불필요하므로 루프로 처리합니다.
#    /code 와의 시그니처 호환을 위해 동일한 인터페이스를 유지합니다.
# ─────────────────────────────────────────────────────────────────────────────

def batch_generate(
    llm: Llama,
    messages_list: List[List[Dict[str, str]]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
    fallback: Optional[str] = None,
) -> List[str]:
    """
    여러 대화를 순차적으로 처리합니다.
    (interactive 모드는 기본적으로 단일 대화이므로 배치 병렬화가 필요 없음)

    /code/utils/llm_utils.py 의 batch_generate 와 동일한 시그니처입니다.
    """
    if not messages_list:
        return []

    MAX_RETRIES = 3
    responses: List[str] = []

    for messages in messages_list:
        result = ""
        for attempt in range(MAX_RETRIES):
            result = generate_response(
                llm, messages,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                stop_at_newline=stop_at_newline,
            )
            if result:
                break
            print(f"[batch_generate] 빈 응답 재시도 (attempt {attempt + 1}/{MAX_RETRIES})")

        if not result:
            if fallback is not None:
                print(f"[batch_generate] WARNING: 재시도 후에도 빈 응답 → fallback 사용: {repr(fallback)}")
                result = fallback
            else:
                raise RuntimeError(
                    f"[batch_generate] {MAX_RETRIES}회 재시도 후에도 응답이 비어 있습니다. "
                    "messages 또는 모델 출력을 확인하세요."
                )
        responses.append(result)

    return responses


# ─────────────────────────────────────────────────────────────────────────────
# 5. 대화 요약 생성
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_SYSTEM = (
    "You are a thorough conversation summarizer. "
    "Summarize ALL meal details discussed so far in 3-5 sentences. "
    "Cover every food item, preparation method, ingredient, and portion that was mentioned. "
    "Be factual and complete — do not omit any detail."
)

_SUMMARIZE_SYSTEM_INCREMENTAL = (
    "You are an incremental conversation summarizer. "
    "You will be given a previous summary and new conversation turns that occurred after it. "
    "Produce an updated summary that incorporates both. "
    "Cover every food item, preparation method, ingredient, and portion mentioned across all turns. "
    "Be factual, complete, and use 3-5 sentences."
)


def summarize_conversation(
    llm: Llama,
    conversation_text: str,
    prev_summary: str = "",
    max_new_tokens: int = 180,
) -> str:
    """
    대화 텍스트를 3-5 문장으로 요약합니다.
    prev_summary 가 있으면 기존 요약 + 신규 턴을 합쳐 증분 업데이트합니다.
    / Summarises the conversation in 3-5 sentences.
      When prev_summary is provided, incrementally updates from the previous summary.
    """
    if prev_summary and conversation_text:
        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_INCREMENTAL},
            {
                "role": "user",
                "content": (
                    f"Previous summary:\n{prev_summary}\n\n"
                    f"New conversation turns since the previous summary:\n\n"
                    f"{conversation_text}\n\n"
                    "Now write the updated summary incorporating all information:"
                ),
            },
        ]
    else:
        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Conversation to summarize:\n\n"
                    f"{conversation_text}\n\n"
                    "Now write a thorough summary of all meal details discussed:"
                ),
            },
        ]
    return generate_response(
        llm,
        messages,
        sampling="greedy",
        max_new_tokens=max_new_tokens,
        stop_at_newline=False,   # 요약은 여러 문장 허용
    )
