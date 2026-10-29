
from langchain.agents.middleware import before_agent, wrap_model_call
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

PREDICTION_KEYWORDS = [
    "예측", "예상", "전망",
    "시뮬레이션", "판매량",
    "광고비", "광고 예산",
    "할인율", "할인",
    "프로모션", "재고",
    "품절", "소진",
    "발주", "수요",
    "몇 주", "몇주",
    "몇 개", "몇개",
]

PREDICTION_EXPLANATION_KEYWORDS = [
    "사용 방법",
    "사용법",
    "어떤 기능",
    "무슨 기능",
    "무엇을 할 수",
    "어떻게 사용",
    "원리가 뭐",
]

OUT_OF_SCOPE_KEYWORDS = [
    "점심", "저녁 메뉴",
    "아침 메뉴", "메뉴 추천",
    "음식 추천", "맛집",
    "날씨", "기온",
    "운세", "로또",
    "영화 추천",
    "노래 추천",
    "파이썬 코드",
    "자바 코드",
]

BLOCK_MESSAGE = (
    "해당 질문은 지원하지 않습니다. "
    "판매량 예측, 광고비·할인율 시뮬레이션, "
    "재고 소진 및 품절 위험 분석과 관련된 "
    "질문을 입력해 주세요."
)


def get_message_content(message):
    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", ""))


def is_user_message(message):
    if isinstance(message, HumanMessage):
        return True

    if isinstance(message, dict):
        return message.get("role") in ("user", "human")

    return getattr(message, "type", None) == "human"


def is_business_question(question):
    question = question.lower()

    return any(
        keyword in question
        for keyword in BUSINESS_KEYWORDS
    )


def is_followup_question(question):
    question = question.lower()

    return any(
        keyword in question
        for keyword in FOLLOWUP_KEYWORDS
    )


def is_prediction_question(question):
    question = question.lower()

    return any(
        keyword in question
        for keyword in PREDICTION_KEYWORDS
    )


def is_prediction_explanation(question):
    question = question.lower()

    return any(
        keyword in question
        for keyword in PREDICTION_EXPLANATION_KEYWORDS
    )


def get_user_questions(state):
    return [
        get_message_content(message).lower().strip()
        for message in state.get("messages", [])
        if is_user_message(message)
    ]


@before_agent(can_jump_to=["end"])
def business_guardrail(state, runtime):
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


@before_agent(can_jump_to=["end"])
def prediction_guardrail(state, runtime):
    questions = get_user_questions(state)

    if not questions:
        return None

    question = questions[-1]
    previous_questions = questions[:-1]

    print("\n[Prediction Guardrail]")
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

    if any(
        keyword in question
        for keyword in OUT_OF_SCOPE_KEYWORDS
    ):
        print("결과: 업무 외 질문 차단\n")
        return {
            "messages": [
                AIMessage(content=BLOCK_MESSAGE)
            ],
            "jump_to": "end",
        }

    if is_prediction_question(question):
        print("결과: 예측 관련 질문 허용\n")
        return None

    if is_prediction_explanation(question):
        print("결과: 예측 기능 설명 허용\n")
        return None

    has_prediction_history = any(
        is_prediction_question(previous_question)
        for previous_question in previous_questions
    )

    if (
        has_prediction_history
        and is_followup_question(question)
    ):
        print("결과: 예측 후속 질문 허용\n")
        return None

    print("결과: 예측 범위 외 질문 차단\n")

    return {
        "messages": [
            AIMessage(content=BLOCK_MESSAGE)
        ],
        "jump_to": "end",
    }


@wrap_model_call
def log_token_usage(request, handler):
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
