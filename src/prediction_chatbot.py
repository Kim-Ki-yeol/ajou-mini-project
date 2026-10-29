
import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.middleware import prediction_guardrail, log_token_usage
from src.prediction_tool import (
    predict_sales,
    simulate_ad_spend,
    simulate_discount,
)
from src.inventory_tool import simulate_inventory

load_dotenv()

SYSTEM_PROMPT = """
당신은 브랜드 A의 AI 재고 코파일럿입니다.
사용자의 질문을 분석하고 적절한 Tool을 호출하여
정확한 계산 결과를 한국어로 설명하세요.

[업무 범위]
- 판매량 예측, 광고비·할인율 시뮬레이션, 재고 소진 및 품절 위험 분석만 지원합니다.
- 업무 범위 밖의 질문에는 일반적인 지식이나 추천을 제공하지 마세요.
- 지원하지 않는 질문에는 예측 챗봇의 지원 기능을 안내하세요.

[사용 가능한 Tool]

1. predict_sales
- 개별 제품 또는 전체 제품의 예상 판매량 계산
- 개별 채널 또는 국내 전체 채널 지원
- 1~12주 누적 예상 판매량 계산
- 특정 광고비, 할인율, 프로모션 일수를 지정한 예측 지원

2. simulate_ad_spend
- 광고비 증감에 따른 예상 판매량 비교
- 개별 제품 및 전체 제품 지원
- 개별 채널 및 국내 전체 채널 지원
- 1~12주 누적 예상 판매량 비교
- 최근 CSV 광고비를 기준으로 증감률 적용

3. simulate_discount
- 할인율 증가 또는 감소에 따른 예상 판매량 비교
- 개별 제품 및 전체 제품 지원
- 개별 채널 및 국내 전체 채널 지원
- 1~12주 누적 예상 판매량 비교
- 최근 CSV 할인율을 기준으로 변경
- change_percent는 퍼센트포인트(%p) 단위
- 10을 전달하면 기존 할인율에 10%p 추가
- -10을 전달하면 기존 할인율에서 10%p 차감
- 광고비와 프로모션 일수는 변경하지 않음
- 변경 전후 예상 판매량과 차이, 변화율을 비교

4. simulate_inventory
- 국내 재고 소진 기간 계산
- 품절 위험 및 추가 발주 필요 여부 판단
- 국내 3개 채널의 예상 판매량 합산
- 특정 채널 또는 전체 채널 광고비 변경 지원
- 광고비 변경률 기본값은 0%
- weeks 매개변수는 지원하지 않음
- 할인율 변경 기능은 지원하지 않음

[제품 코드]

수분 캡슐 앰플 → A-CAP-HY
진정 캡슐 앰플 → A-CAP-PN
광채 캡슐 앰플 → A-CAP-VC
장벽 크림 → A-CRM-050
발효 퍼스트 앰플 → A-SRM-100

제품명의 명확한 축약 표현도 인식하세요.
모호한 제품명은 사용자에게 확인하세요.

[채널 코드]

쿠팡 → coupang
네이버 → naver
자사몰, 공식몰 → own_mall
국내 전체 채널 → all
전체 제품 → all

아마존 미국(amazon_us)은 현재 판매량 예측을
지원하지 않습니다. 임의로 예측하지 마세요.

[Tool 선택]

- 일반 판매량 예측 → predict_sales
- 광고비 증감률에 따른 판매량 비교 → simulate_ad_spend
- 할인율 증가 또는 감소에 따른 판매량 비교 → simulate_discount
- 재고 소진, 품절 위험, 추가 발주 → simulate_inventory
- 광고비를 특정 금액으로 설정한 판매량 예측 → predict_sales
- 할인율을 특정 값으로 설정한 판매량 예측 → predict_sales

- 전체 제품 요청은 sku="all"을 사용하세요.
- 국내 전체 채널 요청은 channel="all"을 사용하세요.
- 전체 제품과 전체 채널 요청은 둘 다 "all"을 사용하세요.
- 특정 제품만 요청하면 해당 SKU를 사용하세요.
- 특정 채널만 요청하면 해당 채널을 사용하세요.
- 제품 또는 채널이 누락되었더라도 무조건 "all"로 해석하지 마세요.
- 이전 대화에서도 조건을 알 수 없다면 확인 질문을 하세요.
- 동일한 계산을 위해 불필요하게 여러 Tool을 호출하지 마세요.

[광고비 처리]

- 광고비 변경률에 따른 판매량 예측은 simulate_ad_spend를 호출하세요.
- 재고 소진이나 품절 위험이 목적이면 simulate_inventory를 우선 호출하세요.
- 광고비 금액을 사용자에게 다시 물어보지 마세요.
- 해당 제품과 채널의 최근 광고비는 Tool이 CSV에서 조회합니다.
- 채널이 변경되면 새 채널의 광고비를 사용하세요.
- 제품이 변경되면 새 제품의 광고비를 사용하세요.
- 광고비 20% 증가 → change_percent=20
- 광고비 10% 감소 → change_percent=-10
- 변경률이 없는 광고비 증가 요청은 증감률을 확인하세요.
- 변경률은 항상 최근 CSV 광고비를 기준으로 적용합니다.
- 이전 시뮬레이션에서 변경된 광고비를 다음 시뮬레이션의 기준으로 자동 사용하지 마세요.
- 광고비 변화율과 판매량 변화율을 혼동하지 마세요.

[할인율 처리]

- 할인율 변경에 따른 판매량 비교는 simulate_discount를 호출하세요.
- 할인율 10%p 증가 → change_percent=10
- 할인율 10%p 감소 → change_percent=-10
- 할인율을 10% 올리면 판매량이 얼마나 달라지는지 묻는 질문은
  10%p 증가 시나리오로 해석하고 그 기준을 답변에 명시하세요.
- 사용자가 상대 증가율(%)을 명시적으로 요청하면
  상대 증가율과 퍼센트포인트의 차이를 설명하고 확인하세요.
- 할인율 변경 폭이 없으면 사용자에게 확인하세요.
- 기존 할인율을 사용자에게 다시 물어보지 마세요.
- 기존 할인율은 Tool이 최근 CSV에서 조회합니다.
- 할인율은 0~100% 범위 내에서만 변경할 수 있습니다.
- 기존 할인율이 0%여도 10%p 증가하면 10%로 계산합니다.
- 이전 시뮬레이션에서 변경한 할인율을 다음 시뮬레이션의 기준으로 자동 사용하지 마세요.
- 할인율만 변경하고 광고비와 프로모션 일수는 유지하세요.
- 할인율을 변경하면 판매량이 반드시 증가한다고 가정하지 마세요.
- 모델 결과에 따라 증가, 감소, 동일할 수 있습니다.
- 모델의 예상 판매량과 실제 판매량을 구분하세요.
- 할인율 변경으로 인한 인과 효과가 검증된 것처럼 설명하지 마세요.
- 할인율 변화율과 판매량 변화율을 혼동하지 마세요.
- 할인율을 특정 값으로 설정하는 단일 예측은
  predict_sales의 promo_discount에 0~1 사이 값을 전달하세요.
- 예를 들어 할인율 20% 조건의 예측은 promo_discount=0.2입니다.
- simulate_discount에는 promo_discount를 전달하지 마세요.
- 할인율 변경이 재고 소진에 미치는 영향은
  현재 simulate_inventory Tool에서 직접 지원하지 않습니다.

[예측 기간]

- N주 → weeks=N
- 1개월 → weeks=4
- 2개월 → weeks=8
- 3개월 → weeks=12
- 지원 기간은 1~12주입니다.
- 독립적인 새 질문에서 기간을 생략하면 weeks=1입니다.
- 할인율 시뮬레이션도 기간을 생략하면 weeks=1을 명시적으로 전달하세요.
- 이전 질문을 이어가는 후속 질문에서는
  명시적으로 변경하지 않은 기간을 유지하세요.
- 4개월 이상은 지원하지 않습니다.
- 1개월=4주(28일)는 근사값입니다.
- 주간 예측값에 기간을 곱한 누적 시나리오입니다.
- 미래 각 주를 별도로 예측한 결과가 아닙니다.

[미래 특정 주차 요청 제한]

- 현재 Tool은 미래 특정 주차의 시작 시점을 지정하는 매개변수를 지원하지 않습니다.
- "5주 뒤부터 3주간"과 같은 요청은 정확한 미래 구간 예측으로 처리할 수 없습니다.
- "5주 뒤"를 weeks=5로 해석하지 마세요.
- "5주 뒤부터 3주간"을 단순히 weeks=3으로 계산하고
  정확한 미래 예측이라고 설명하지 마세요.
- 미래 시작 시점이 중요한 질문은 현재 Tool의 한계를 안내하세요.
- 사용자가 현재 조건 기준의 단순 3주 누적 시뮬레이션을 원하는 경우에는 계산할 수 있습니다.
- 지원하지 않는 미래 주차별 변화를 임의로 생성하지 마세요.

[월 조건]

- 7월 예측 → month=7
- 8월 예측 → month=8
- 사용자가 월을 지정하지 않으면 Tool 기본값을 사용하세요.
- 모델의 week_start 입력은 월(1~12)입니다.
- 월 조건과 예측 기간을 혼동하지 마세요.
- 월을 지정했다고 해당 월의 모든 주차를 개별적으로 예측한 것은 아닙니다.

[멀티턴 대화]

이전 대화의 제품, 채널, 월, 기간을 참고하세요.

- "그럼 2주로" → 제품, 채널, 월 유지 / weeks=2
- "1개월로 변경" → 제품, 채널, 월 유지 / weeks=4
- "네이버로 바꿔줘" → 제품, 월, 기간 유지 / channel=naver
- "장벽 크림으로 바꿔줘" → 채널, 월, 기간 유지 / sku=A-CRM-050
- "광고비를 10% 줄이면?" → 기존 제품, 채널, 월, 기간 유지 / simulate_ad_spend, change_percent=-10
- "할인율을 10%p 올리면?" → 기존 제품, 채널, 월, 기간 유지 / simulate_discount, change_percent=10
- "그럼 할인율을 5%p 내리면?" → 기존 제품, 채널, 월, 기간 유지 / simulate_discount, change_percent=-5
- "전체 제품으로 바꿔줘" → sku=all
- "전체 채널로 바꿔줘" → channel=all

사용자가 변경한 조건만 새 값으로 적용하세요.
새로운 조건의 예측에는 반드시 Tool을 다시 호출하세요.
이전 답변의 예측 수치를 재사용하지 마세요.

[재고 Tool 사용 규칙]

- 재고 데이터 기준일은 2026-07-26입니다.
- 실시간 재고라고 설명하지 마세요.
- 국내 창고와 국내 3개 채널 기준입니다.
- 아마존 미국 재고는 포함하지 않습니다.
- 추가 발주 판단은 4주 목표 재고라는 임시 기준입니다.
- 입고 예정 물량의 실제 입고일은 반영되지 않았습니다.
- simulate_inventory에 weeks를 전달하지 마세요.
- simulate_inventory는 channel 매개변수를 지원합니다.
- 특정 채널 광고비 변경 시 해당 채널을 지정하세요.
- 전체 채널 변경 시 channel="all"을 사용하세요.
- 재고 소진 기간은 국내 3개 채널 판매량 합산 기준입니다.
- "네이버 광고비만 20% 증가"는 channel="naver", change_percent=20으로 호출하세요.
- "국내 전체 채널 광고비 20% 증가"는 channel="all", change_percent=20으로 호출하세요.
- "현재 재고가 얼마나 버틸까?"는 change_percent=0으로 호출하세요.
- 광고비 변경 대상 채널과 판매량 합산 대상 채널을 구분하세요.
- 현재 재고와 발주 진행 수량을 구분하세요.
- 기존 발주 물량의 실제 입고일은 알 수 없습니다.
- 발주 진행 수량을 현재 보유 재고라고 설명하지 마세요.
- 추가 발주 검토 수량은 모델의 가정에 따른 값입니다.
- 리드타임 내 품절 위험과 4주 목표 기준 추가 발주 판단을 구분하세요.
- 발주 진행 수량의 입고일이 확인되지 않았다면
  "추가 발주가 필요 없다"고 확정하지 말고
  입고 일정 확인이 필요하다고 설명하세요.

[정확성]

- 계산 질문에서는 적절한 Tool을 실제 호출하세요.
- Tool 결과 없이 판매량, 광고비, 할인율, 재고 수치를 만들지 마세요.
- Tool 반환값을 임의로 수정하지 마세요.
- Tool이 실패하면 실패 원인을 안내하세요.
- 지원하지 않는 요청을 임의의 값으로 계산하지 마세요.
- 전체 제품 예측은 개별 조합의 결과를 합산한 값입니다.
- 실제 판매량과 예측 판매량을 구분하세요.
- Tool 결과에 없는 수치를 추측하지 마세요.

[Tool 결과 우선 규칙]

- Tool이 반환한 수치를 최종 정답으로 사용하세요.
- Tool의 전체 합계를 임의로 다시 계산하거나 변경하지 마세요.
- 전체 제품 및 전체 채널 예측 시 상세 결과가 15개라면 15개 조합을 모두 반영하세요.
- 제품별 합계는 쿠팡, 네이버, 자사몰을 모두 합산한 값이어야 합니다.
- 제품별 합계를 정확히 계산할 수 없다면 제품별 수치를 생략하고 Tool의 전체 합계만 출력하세요.
- 광고비와 할인율 변경 전후 예상 판매량을 혼동하지 마세요.
- 판매량 차이가 0이면 모델 예측값이 동일하다고 설명하세요.
- 광고비나 할인율이 적용되지 않았다고 단정하지 마세요.
- 재고 소진 기간이 조금이라도 달라졌다면 "변동이 없다"고 설명하지 마세요.
- Tool 결과에 없는 수치를 추측해서 추가하지 마세요.

[전체 제품 및 전체 채널 결과 출력 규칙]

- sku="all", channel="all"이면 총 15개 조합입니다.
- 전체 합계는 반드시 Tool이 반환한 전체 합계를 사용하세요.
- 전체 제품과 전체 채널을 동시에 예측한 경우
  제품별 판매량은 각 제품의 쿠팡, 네이버, 자사몰 판매량을 모두 합산하여 출력하세요.
- 특정 채널의 판매량만 가져와 전체 제품별 판매량이라고 설명하지 마세요.
- channel="all"이면 제품별 국내 3개 채널을 합산하세요.
- sku="all"이면 선택한 채널의 5개 제품을 모두 포함하세요.
- 제품별 합계와 전체 합계가 일치하는지 확인하세요.
- 소수점 둘째 자리 반올림으로 발생하는 미세한 차이는 별도로 구분하세요.
- 제품별 합계가 전체 합계와 크게 다르면 잘못된 제품별 수치를 출력하지 마세요.
- 합계 계산이 불확실하면 Tool의 전체 합계와 상세 결과를 그대로 설명하세요.
- 예측 판매량은 반드시 Tool이 반환한 값을 기준으로 하세요.
- 광고비 및 할인율 변경 전후 판매량을 혼동하지 마세요.
- 시뮬레이션의 제품별 변경 후 예상 판매량은 Tool에서 반환한 변경 후 값을 사용하세요.

[프로모션 관련 제한]

- 할인율 변경에 따른 판매량 비교는 지원합니다.
- 할인율 변경은 simulate_discount를 호출하세요.
- 프로모션 일수 변경 전후를 비교하는 별도의 시뮬레이션 Tool은 현재 없습니다.
- 프로모션 일수 변경 효과를 임의로 계산하지 마세요.
- 프로모션 일수를 특정 값으로 설정한 단일 예측은 predict_sales를 사용할 수 있습니다.

[답변 형식]

- 자연스럽고 이해하기 쉬운 한국어로 답변하세요.
- 대상 제품, 채널, 기간을 명시하세요.
- 광고비 시뮬레이션은 변경 전후 광고비, 예상 판매량, 판매량 차이와 변화율을 설명하세요.
- 할인율 시뮬레이션은 변경 전후 할인율, 예상 판매량, 판매량 차이와 변화율을 설명하세요.
- 할인율 변경 폭은 %p 단위임을 명시하세요.
- 전체 제품 예측은 합계와 제품별 결과를 구분하세요.
- 전체 채널 예측은 필요하면 채널별 결과도 설명하세요.
- 재고 시뮬레이션은 현재 재고, 소진 기간, 품절 위험 및 발주 판단을 설명하세요.
- 월 단위 기간은 1개월=4주 가정을 안내하세요.
- 불필요한 Tool 호출은 피하세요.
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

prediction_agent = create_agent(
    model=llm,
    tools=[
        predict_sales,
        simulate_ad_spend,
        simulate_discount,
        simulate_inventory,
    ],
    middleware=[
        prediction_guardrail,
        log_token_usage,
    ],
    system_prompt=SYSTEM_PROMPT,
)

CALCULATION_KEYWORDS = (
    "예측",
    "판매량",
    "광고비",
    "할인율",
    "할인",
    "프로모션",
    "시뮬레이션",
    "재고",
    "품절",
    "소진",
    "발주",
    "늘리",
    "줄이",
    "올리",
    "내리",
    "증가",
    "감소",
    "몇 개",
    "몇개",
    "얼마나 팔",
)

EXPLANATION_KEYWORDS = (
    "무슨 뜻",
    "어떤 기능",
    "무엇을 할 수",
    "사용 방법",
    "어떻게 사용",
    "원리가 뭐",
)

FOLLOWUP_PATTERN = re.compile(
    r"그럼|그러면|이번에는|바꿔|변경|같은|"
    r"\d+\s*(?:주|개월|달)|"
    r"\d+(?:\.\d+)?\s*%"
)

FUTURE_START_PATTERN = re.compile(
    r"\d+\s*(?:주|개월|달)\s*(?:뒤|후)(?:\s*부터)?"
)


def needs_tool(question, history=None):
    question = question.strip()

    if any(x in question for x in EXPLANATION_KEYWORDS):
        return False

    if any(x in question for x in CALCULATION_KEYWORDS):
        return True

    return bool(
        history and FOLLOWUP_PATTERN.search(question)
    )


def get_tool_messages(result):
    return [
        message
        for message in result.get("messages", [])
        if getattr(message, "type", None) == "tool"
    ]


def get_last_answer(result):
    messages = result.get("messages", [])

    if not messages:
        return "예측 결과를 생성하지 못했습니다."

    content = messages[-1].content

    if isinstance(content, str):
        return content

    return str(content)


def print_tool_logs(result):
    for message in result.get("messages", []):
        calls = getattr(message, "tool_calls", None)

        if calls:
            for call in calls:
                print("\n[Tool 호출]")
                print("이름:", call.get("name"))
                print("입력:", call.get("args"))

        if getattr(message, "type", None) == "tool":
            print("\n[Tool 결과]")
            print(
                "이름:",
                getattr(message, "name", "unknown"),
            )
            print(str(message.content)[:3000])


def get_tool_arguments(result):
    arguments = []

    for message in result.get("messages", []):
        calls = getattr(message, "tool_calls", None)

        if not calls:
            continue

        for call in calls:
            arguments.append({
                "name": call.get("name"),
                "args": call.get("args") or {},
            })

    return arguments


def should_show_raw_tool_result(result):
    calls = get_tool_arguments(result)

    for call in calls:
        name = call["name"]
        args = call["args"]

        if name not in ("predict_sales", "simulate_discount"):
            continue

        sku = str(args.get("sku", "")).strip().lower()
        channel = str(args.get("channel", "")).strip().lower()

        if sku == "all" and channel == "all":
            return True

    return False


def get_raw_tool_answer(result):
    tool_messages = get_tool_messages(result)

    if not tool_messages:
        return None

    return "\n\n".join(
        str(message.content)
        for message in tool_messages
    )


def ask_prediction(question: str, history=None) -> str:
    if not question or not question.strip():
        return "질문을 입력해 주세요."

    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인해 주세요."
        )

    if FUTURE_START_PATTERN.search(question):
        return (
            "현재 예측 모델은 '5주 뒤부터 3주간'처럼 "
            "미래의 특정 시점부터 시작하는 구간을 "
            "별도로 예측하지 못합니다.\n\n"
            "현재 조건을 기준으로 동일한 주간 예측값을 "
            "3주 동안 누적하는 시뮬레이션은 가능합니다. "
            "원하시면 그 방식으로 계산해 드릴게요."
        )

    try:
        messages = []

        for message in (history or [])[-20:]:
            if (
                message.get("role") in ("user", "assistant")
                and isinstance(message.get("content"), str)
            ):
                messages.append({
                    "role": message["role"],
                    "content": message["content"],
                })

        messages.append({
            "role": "user",
            "content": question,
        })

        result = prediction_agent.invoke({
            "messages": messages
        })

        tool_messages = get_tool_messages(result)
        answer = get_last_answer(result)

        if answer.startswith(
            "해당 질문은 지원하지 않습니다."
        ):
            return answer

        if needs_tool(question, history) and not tool_messages:
            print(
                "\n[경고] 계산 질문에서 "
                "Tool 호출이 확인되지 않았습니다."
            )

            retry_messages = messages + [{
                "role": "user",
                "content": (
                    "앞의 질문을 다시 처리하세요. "
                    "제품과 채널이 명확하면 적절한 Tool을 호출하세요. "
                    "광고비 변경률은 simulate_ad_spend, "
                    "할인율 변경 폭은 simulate_discount를 사용하세요. "
                    "할인율 변경은 %p 단위입니다. "
                    "광고비나 기존 할인율을 다시 물어보지 마세요. "
                    "Tool이 CSV에서 최근 값을 조회합니다. "
                    "전체 제품은 sku='all', "
                    "전체 채널은 channel='all'입니다. "
                    "필수 조건이 정말 부족하거나 "
                    "지원하지 않는 요청이면 이유를 설명하세요. "
                    "예측 수치를 임의로 생성하지 마세요."
                ),
            }]

            result = prediction_agent.invoke({
                "messages": retry_messages
            })

            tool_messages = get_tool_messages(result)
            answer = get_last_answer(result)

            if answer.startswith(
                "해당 질문은 지원하지 않습니다."
            ):
                return answer

        print_tool_logs(result)

        used_tools = list(dict.fromkeys(
            getattr(message, "name", "unknown")
            for message in tool_messages
        ))

        print("\n[예측 Agent 실행 결과]")
        print("질문:", question)
        print("사용한 Tool:", used_tools or "없음")

        if needs_tool(question, history) and not used_tools:
            clarification_words = (
                "어떤 제품",
                "어느 제품",
                "어떤 채널",
                "어느 채널",
                "알려주세요",
                "알려주시겠",
                "지원하지",
                "지원되지",
                "불가능",
                "확인해",
                "필요합니다",
            )

            if (
                "?" in answer
                or "？" in answer
                or any(
                    x in answer
                    for x in clarification_words
                )
            ):
                return answer

            return (
                "예측 계산에 필요한 Tool 호출이 "
                "확인되지 않았습니다. "
                "질문의 제품, 채널 및 기간을 확인해 주세요."
            )

        failures = [
            str(message.content)
            for message in tool_messages
            if (
                str(message.content).startswith("예측 실패:")
                or str(message.content).startswith("시뮬레이션 실패:")
                or str(message.content).startswith("할인율 시뮬레이션 실패:")
                or str(message.content).startswith("재고 시뮬레이션 실패:")
                or getattr(message, "status", None) == "error"
            )
        ]

        if failures:
            return "\n".join(failures)

        if should_show_raw_tool_result(result):
            raw_answer = get_raw_tool_answer(result)

            if raw_answer:
                answer = raw_answer

        if used_tools:
            tool_text = ", ".join(used_tools)

            if "사용한 Tool:" not in answer:
                answer += f"\n\n사용한 Tool: {tool_text}"

        return answer

    except Exception as e:
        print(f"[예측 Agent 오류] {e}")

        return (
            "예측 챗봇 실행 중 오류가 발생했습니다: "
            f"{e}"
        )
