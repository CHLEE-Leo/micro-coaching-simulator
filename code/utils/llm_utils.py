"""
utils/llm_utils.py
──────────────────
vLLM 기반 LLM 공통 유틸리티.
  - 모델 로딩 (vLLM LLM 객체, 토크나이저 불필요)
  - SamplingParams 빌더
  - 단일 응답 생성
  - 배치 응답 생성  ← 병렬 시뮬레이션의 핵심
  - 대화 요약 생성

vLLM 을 사용하면:
  1. 모델 로딩 후 동일 객체로 배치 추론 가능 (torch.no_grad, device_map 관리 불필요)
  2. tensor_parallel_size 를 지정해 멀티 GPU 에 자동 분산 (pipeline parallel 대신 tensor parallel)
  3. batch_generate() 로 N 개의 대화를 한 번에 GPU 에 밀어넣어 처리량 극대화
"""

from __future__ import annotations

from typing import List, Dict, Optional

from vllm import LLM, SamplingParams


# ──────────────────────────────────────────────────────────────────────────────
# 1. 모델 로딩
# ──────────────────────────────────────────────────────────────────────────────

def load_model(
    llm_repo: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    dtype: str = "bfloat16",
) -> LLM:
    """
    vLLM LLM 객체를 로드합니다. 토크나이저는 vLLM 내부에서 자동 관리됩니다.

    Parameters
    ----------
    llm_repo             : 'meta-llama/Llama-3.3-70B-Instruct' 등 HuggingFace repo 경로
    tensor_parallel_size : 텐서 병렬 GPU 수 (GPU 수에 맞게 설정, 기본 1)
    max_model_len        : 최대 컨텍스트 길이 (KV 캐시 메모리 절감 목적으로 줄일 수 있음)
    dtype                : 가중치 데이터 타입 ('float16' | 'bfloat16' | 'auto')

    Returns
    -------
    vllm.LLM
    """
    return LLM(
        model=llm_repo,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. SamplingParams 빌더
# ──────────────────────────────────────────────────────────────────────────────

def build_sampling_params(
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
) -> SamplingParams:
    """
    생성 전략에 따라 vLLM SamplingParams 를 반환합니다.

    Parameters
    ----------
    sampling        : "greedy" | "sampling"
                      (vLLM 은 beam search 를 지원하지 않으므로 "beam" 은 greedy 로 처리)
    max_new_tokens  : 최대 생성 토큰 수
    stop_at_newline : True 이면 post-processing 에서 첫 번째 비어 있지 않은 줄만 취합니다.
                      stop 시퀀스로 "\n" 을 쓰지 않는 이유:
                      Gemma 등 일부 모델의 chat template 이 assistant 응답 직전에 "\n" 을
                      삽입하므로, "\n" 을 stop 토큰으로 지정하면 vLLM 이 즉시 멈춰
                      빈 문자열을 반환합니다. 개행 처리는 post-processing 으로만 합니다.
    """
    if sampling in ("beam", "greedy"):
        return SamplingParams(
            temperature=0.0,   # greedy
            max_tokens=max_new_tokens,
            repetition_penalty=1.15,   # 반복 표현 억제
        )
    elif sampling == "sampling":
        return SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_new_tokens,
            repetition_penalty=1.15,   # 반복 표현 억제
        )
    else:
        raise ValueError(
            f"지원하지 않는 sampling 방식: '{sampling}'. "
            "beam | greedy | sampling 중 선택하세요."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. 단일 응답 생성
# ──────────────────────────────────────────────────────────────────────────────

def generate_response(
    llm: LLM,
    messages: List[Dict[str, str]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
) -> str:
    """
    단일 messages 리스트에 대해 응답을 생성합니다.
    내부적으로 batch_generate 를 1-배치로 호출합니다.

    Parameters
    ----------
    llm            : vLLM LLM 객체
    messages       : [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    sampling       : "beam" | "greedy" | "sampling"
    max_new_tokens : 최대 생성 토큰 수
    stop_at_newline: 개행 시 조기 중단 여부 (발화 생성=True, 요약=False)

    Returns
    -------
    str : 생성된 응답 텍스트
    """
    results = batch_generate(
        llm,
        [messages],
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        stop_at_newline=stop_at_newline,
    )
    return results[0]


# ──────────────────────────────────────────────────────────────────────────────
# 4. 배치 응답 생성  ← 병렬 시뮬레이션의 핵심
# ──────────────────────────────────────────────────────────────────────────────

def batch_generate(
    llm: LLM,
    messages_list: List[List[Dict[str, str]]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
    fallback: Optional[str] = None,
) -> List[str]:
    """
    여러 대화를 vLLM 에 한 번에 넘겨 배치로 생성합니다.

    vLLM 은 내부적으로 continuous batching 을 수행하므로
    N 개의 요청을 리스트로 넘기면 GPU 를 최대한 활용해 병렬 처리합니다.

    Parameters
    ----------
    llm           : vLLM LLM 객체
    messages_list : [conv_1_messages, conv_2_messages, ...]
                    각 element → [{"role": ..., "content": ...}, ...]
    sampling      : "beam" | "greedy" | "sampling"
    max_new_tokens: 각 응답의 최대 토큰 수
    stop_at_newline: 개행에서 조기 중단 여부

    Returns
    -------
    List[str] : 각 대화에 대응하는 생성 텍스트 목록 (순서 보존)
    """
    if not messages_list:
        return []

    sp = build_sampling_params(
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        stop_at_newline=stop_at_newline,
    )

    # llm.chat() 은 List[List[dict]] 형식을 직접 받아 배치 처리
    # 빈 응답이 나온 요청은 최대 MAX_RETRIES 번까지 재시도합니다.
    MAX_RETRIES = 3

    def _extract(raw: str) -> str:
        if stop_at_newline:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            return lines[0] if lines else ""
        return raw.strip()

    outputs = llm.chat(messages_list, sampling_params=sp, use_tqdm=False)
    responses: List[str] = [_extract(out.outputs[0].text) for out in outputs]

    for attempt in range(1, MAX_RETRIES + 1):
        # 빈 응답이 남아 있는 인덱스만 추림
        empty_indices = [i for i, r in enumerate(responses) if not r]
        if not empty_indices:
            break
        print(f"[batch_generate] 빈 응답 {len(empty_indices)}건 재시도 (attempt {attempt}/{MAX_RETRIES})")
        retry_msgs  = [messages_list[i] for i in empty_indices]
        retry_outs  = llm.chat(retry_msgs, sampling_params=sp, use_tqdm=False)
        for idx, out in zip(empty_indices, retry_outs):
            responses[idx] = _extract(out.outputs[0].text)

    # 재시도 후에도 여전히 비어 있으면:
    #   fallback 이 지정된 경우 → 경고만 출력하고 fallback 으로 채워 계속 진행
    #   fallback 이 None 인 경우  → RuntimeError (원천 봉쇄)
    still_empty = [i for i, r in enumerate(responses) if not r]
    if still_empty:
        if fallback is not None:
            print(
                f"[batch_generate] WARNING: {MAX_RETRIES}회 재시도 후에도 "
                f"idx={still_empty} 응답이 비어 있음 → fallback 사용: {repr(fallback)}"
            )
            for i in still_empty:
                responses[i] = fallback
        else:
            raise RuntimeError(
                f"[batch_generate] {MAX_RETRIES}회 재시도 후에도 "
                f"idx={still_empty} 의 응답이 여전히 비어 있습니다. "
                f"messages_list 또는 모델 출력을 확인하세요."
            )

    return responses


# ──────────────────────────────────────────────────────────────────────────────
# 5. 대화 요약 생성 (Principle 4)
# ──────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_SYSTEM = (
    "You are a concise conversation summarizer. "
    "Summarize the key meal information discussed so far in 2–3 sentences. "
    "Focus only on what food items, preparation methods, ingredients, and portions have been mentioned. "
    "Be factual and brief."
)


def summarize_conversation(
    llm: LLM,
    conversation_text: str,
    max_new_tokens: int = 120,
) -> str:
    """
    지금까지의 대화 텍스트를 2–3 문장으로 요약합니다.
    생성된 요약은 Coach·User 의 시스템 프롬프트에 전역 메모리로 주입됩니다.

    Parameters
    ----------
    llm               : 요약에 사용할 vLLM LLM 객체 (Coach 와 공유 가능)
    conversation_text : "Coach: ...\nUser: ...\n..." 형식의 대화 문자열
    max_new_tokens    : 요약 최대 생성 토큰 수

    Returns
    -------
    str : 요약 문자열
    """
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Conversation to summarize:\n\n"
                f"{conversation_text}\n\n"
                "Now write a 2–3 sentence summary:"
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
