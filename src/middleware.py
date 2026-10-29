from langchain.agents.middleware import (
    before_agent,
    wrap_model_call,
)
from langchain_core.messages import AIMessage, HumanMessage


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


FOLLOWUP_KEYWORDS = [
    "그럼", "그러면",
    "이번에는", "이번엔",
    "그 제품", "그 상품",
    "그 채널", "같은 제품",
    "같은 채널",
    "아까", "방금",
    "이전", "앞에서",
    "다시", "비교",
    "더", "줄여", "늘려",
    "그중", "그 중",
    "동일한", "똑같이",
    "그것", "그거",
    "얼마나", "몇 개",
    "몇개", "어때",
]


def get_message_content(message):
    """LangChain 메시지와 dict 메시지를 모두 처리한다."""

    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", ""))


def is_user_message(message):
    """사용자 메시지 여부를 확인한다."""

    if isinstance(message, HumanMessage):
        return True

    if isinstance(message, dict):
        return message.get("role") in ("user", "human")

    return getattr(message, "type", None) == "human"


def is_business_question(question):
    """업무 관련 키워드가 포함되어 있는지 확인한다."""

    question = question.lower()

    return any(
        keyword in question
        for keyword in BUSINESS_KEYWORDS
    )


def is_followup_question(question):
    """이전 질문에 이어지는 표현인지 확인한다."""

    question = question.lower()

    return any(
        keyword in question
        for keyword in FOLLOWUP_KEYWORDS
    )


@before_agent(can_jump_to=["end"])
def business_guardrail(state, runtime):
    """
    업무 관련 질문을 허용하고,
    이전 업무 대화에 이어지는 후속 질문도 허용한다.
    """

    messages = state.get("messages", [])

    if not messages:
        return None

    user_messages = [
        message
        for message in messages
        if is_user_message(message)
    ]

    if not user_messages:
        return None

    question = get_message_content(
        user_messages[-1]
    ).lower().strip()

    previous_questions = [
        get_message_content(message).lower().strip()
        for message in user_messages[:-1]
    ]

    print("\n[Business Guardrail]")
    print("현재 질문:", question)
    print("이전 질문:", previous_questions)

    if question in [
        "안녕",
        "안녕하세요",
        "고마워",
        "감사합니다",
    ]:
        print("결과: 일반 인사 허용\n")
        return None

    if is_business_question(question):
        print("결과: 업무 질문 허용\n")
        return None

    has_business_history = any(
        is_business_question(previous_question)
        for previous_question in previous_questions
    )

    is_followup = is_followup_question(question)

    print("업무 대화 기록 존재:", has_business_history)
    print("후속 질문 여부:", is_followup)

    if has_business_history and is_followup:
        print("결과: 이전 업무 대화 후속 질문 허용\n")
        return None

    print("결과: 업무 외 질문 차단\n")

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
        print("")

    return response
