
from langchain.agents.middleware import (
    before_agent,
    wrap_model_call,
)
from langchain_core.messages import AIMessage


BUSINESS_KEYWORDS = [
    "광고비", "광고 예산",
    "판매량", "매출", "재고",
    "품절", "발주", "수요",
    "제품", "상품", "채널",
    "프로모션", "할인",
    "예측", "시뮬레이션",
    "네이버", "쿠팡", "아마존",
    "자사몰", "sku",
]
@before_agent(can_jump_to=["end"])
def business_guardrail(state, runtime):
    """업무와 관계없는 질문을 LLM 호출 전에 차단한다."""

    messages = state.get("messages", [])

    if not messages:
        return None

    question = str(messages[-1].content).lower().strip()

    if question in ["안녕", "안녕하세요", "고마워", "감사합니다"]:
        return None

    if any(word in question for word in BUSINESS_KEYWORDS):
        print("[Middleware] 업무 질문 허용")
        return None

    print("[Middleware] 업무 외 질문 차단")

    return {
        "messages": [
            AIMessage(
                content=(
                    "해당 질문은 지원하지 않습니다. "
                    "광고비, 판매량, 재고, 발주와 관련된 "
                    "질문을 입력해 주세요."
                )
            )
        ],
        "jump_to": "end",
    }
@wrap_model_call
def log_token_usage(request, handler):
    """LLM 호출 후 토큰 사용량을 출력한다."""

    response = handler(request)

    usage = getattr(
        response.result,
        "usage_metadata",
        None
    )

    if usage:
        print("\n[토큰 사용량]")
        print("입력:", usage.get("input_tokens", 0))
        print("출력:", usage.get("output_tokens", 0))
        print("합계:", usage.get("total_tokens", 0))

    return response
