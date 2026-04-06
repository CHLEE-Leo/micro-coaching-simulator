"""
code_interactive/utils/llm_chatgpt.py
──────────────────────────────────────
LangGraph + ChatGPT API 기반 LLM 유틸리티.
/ LLM utility powered by LangGraph + ChatGPT API.

llm_utils.py (llama-cpp-python) 와 동일한 인터페이스를 제공합니다.
  load_model(model_name)   → ChatGPTClient
  generate_response(client, messages, ...) → str
  batch_generate(client, messages_list, ...) → List[str]

session_manager 가 llm_provider 값에 따라 이 모듈 또는 llm_utils.py 를
선택적으로 사용합니다.

/ Provides the same interface as llm_utils.py (llama-cpp-python):
  load_model(model_name)   → ChatGPTClient
  generate_response(client, messages, ...) → str
  batch_generate(client, messages_list, ...) → List[str]
  session_manager picks this or llm_utils.py based on llm_provider.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# .env 로드 — 프로젝트 루트의 .env 에서 OPENAI_API_KEY 를 읽습니다.
# ─────────────────────────────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # code_interactive/utils/
    "..", "..",                                    # micro-coaching-simulator/
    ".env",
)
load_dotenv(os.path.normpath(_ENV_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# ChatGPT 클라이언트 싱글턴 (세션 매니저가 생성)
# ─────────────────────────────────────────────────────────────────────────────

class ChatGPTClient:
    """
    LangGraph 기반 ChatGPT 래퍼.
    load_model() 이 반환하는 객체이며, generate_response() 에 전달됩니다.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature_greedy: float = 0.0,
        temperature_sampling: float = 0.7,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your-"):
            raise ValueError(
                "OPENAI_API_KEY 가 설정되지 않았습니다. "
                ".env 파일에 유효한 API 키를 입력하세요."
            )

        self._model_name = model_name
        self._llm_greedy = ChatOpenAI(
            model=model_name,
            temperature=temperature_greedy,
            api_key=api_key,
        )
        self._llm_sampling = ChatOpenAI(
            model=model_name,
            temperature=temperature_sampling,
            top_p=0.9,
            api_key=api_key,
        )

        # LangGraph 워크플로우 빌드
        self._graph_greedy = self._build_graph(self._llm_greedy)
        self._graph_sampling = self._build_graph(self._llm_sampling)

    @staticmethod
    def _build_graph(llm: ChatOpenAI):
        """단일 LLM 호출 LangGraph 워크플로우."""

        class GraphState(TypedDict):
            messages: List[Dict[str, str]]
            max_tokens: int
            response: str

        def call_llm(state: GraphState) -> GraphState:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lc_messages = []
            for m in state["messages"]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            try:
                result = llm.invoke(lc_messages, max_tokens=state.get("max_tokens", 80))
                return {**state, "response": result.content}
            except Exception as e:
                print(f"[ChatGPT] ERROR in call_llm: {e}")
                return {**state, "response": f"[API_ERROR: {str(e)[:200]}]"}

        builder = StateGraph(GraphState)
        builder.add_node("llm", call_llm)
        builder.add_edge(START, "llm")
        builder.add_edge("llm", END)
        return builder.compile()

    def invoke(
        self,
        messages: List[Dict[str, str]],
        sampling: str = "greedy",
        max_tokens: int = 80,
    ) -> str:
        """LangGraph 를 통해 ChatGPT 에 messages 를 전달하고 응답을 반환합니다."""
        graph = self._graph_greedy if sampling in ("greedy", "beam") else self._graph_sampling
        try:
            result = graph.invoke({
                "messages": messages,
                "max_tokens": max_tokens,
                "response": "",
            })
            resp = result.get("response", "")
            if not resp or not resp.strip():
                print(f"[ChatGPT] WARNING: empty response (model={self._model_name}, tokens={max_tokens})")
            return resp
        except Exception as e:
            print(f"[ChatGPT] ERROR in invoke: {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# llm_utils.py 호환 인터페이스
# / Interface compatible with llm_utils.py
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = "gpt-4.1") -> ChatGPTClient:
    """
    ChatGPT 클라이언트를 생성합니다.
    llm_utils.py 의 load_model() 과 동일한 이름입니다.

    Parameters
    ----------
    model_name : OpenAI 모델명 (기본 "gpt-4.1")

    Returns
    -------
    ChatGPTClient
    """
    return ChatGPTClient(model_name=model_name)


# 하위 호환 별칭 / backward-compatible alias
load_chatgpt_model = load_model


def generate_response(
    client: ChatGPTClient,
    messages: List[Dict[str, str]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
) -> str:
    """
    ChatGPT 를 통해 단일 응답을 생성합니다.
    llm_utils.py 의 generate_response() 와 동일한 이름·시그니처입니다.

    Parameters
    ----------
    client         : ChatGPTClient 객체 (load_model() 반환값)
    messages       : [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    sampling       : "beam" | "greedy" | "sampling"
    max_new_tokens : 최대 생성 토큰 수
    stop_at_newline: True 이면 첫 번째 비어 있지 않은 줄만 반환

    Returns
    -------
    str : 생성된 응답 텍스트
    """
    raw = client.invoke(messages, sampling=sampling, max_tokens=max_new_tokens)

    if stop_at_newline:
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        return lines[0] if lines else ""
    return raw.strip()


# 하위 호환 별칭 / backward-compatible alias
generate_response_chatgpt = generate_response


def batch_generate(
    client: ChatGPTClient,
    messages_list: List[List[Dict[str, str]]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
    fallback: Optional[str] = None,
) -> List[str]:
    """
    여러 대화를 순차적으로 처리합니다.
    llm_utils.py 의 batch_generate() 와 동일한 이름·시그니처입니다.

    / Sequentially processes multiple conversations.
      Same name and signature as llm_utils.py's batch_generate().
    """
    if not messages_list:
        return []
    results: List[str] = []
    for messages in messages_list:
        resp = generate_response(
            client, messages,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            stop_at_newline=stop_at_newline,
        )
        results.append(resp if resp else (fallback or ""))
    return results
