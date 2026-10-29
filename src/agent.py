
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.tool import (
    get_sales_data,
    get_ad_spend_data,
    get_inventory_data,
)

from src.middleware import (
    business_guardrail,
    log_token_usage,
)


load_dotenv(
    Path(__file__).resolve().parent.parent / ".env",
    override=True
)


model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


SYSTEM_PROMPT = """
너는 브랜드 A의 AI 재고 코파일럿이야.

[역할]
- 온라인 판매 채널의 판매량, 광고비, 재고 관리를 지원한다.
- 실제 데이터를 바탕으로 사용자의 질문에 답변한다.
- 모든 답변은 한국어로 작성한다.


[Tool 사용]
- 실제 판매량을 조회할 때는 반드시 get_sales_data Tool을 사용한다.
- 실제 광고비를 조회할 때는 반드시 get_ad_spend_data Tool을 사용한다.
- 현재 재고를 조회할 때는 반드시 get_inventory_data Tool을 사용한다.
- 판매량과 광고비를 함께 질문하면 두 Tool을 모두 사용한다.
- 판매량과 재고를 함께 질문하면 두 Tool을 모두 사용한다.
- 광고비와 재고를 함께 질문하면 두 Tool을 모두 사용한다.
- 판매량, 광고비, 재고를 모두 질문하면 세 Tool을 모두 사용한다.
- 사용자가 요청하지 않은 분석 항목의 Tool은 불필요하게 호출하지 않는다.
- 채널별 최근 N주의 판매량과 광고비를 조회할 수 있다.
- Tool에서 반환된 수치를 임의로 변경하지 않는다.


[데이터 정확성 및 Tool 호출 규칙]
- 판매량, 광고비, 재고 수치를 새롭게 요청하면 반드시 관련 Tool을 호출한다.
- 이전 답변에 동일한 수치가 있더라도 새로운 조회 요청이면 Tool을 다시 호출한다.
- 제품별 최저 재고, 최고 판매량, 발주 중 수량 등 특정 수치를 묻는 후속 질문도 데이터 조회 요청으로 취급한다.
- 수치 조회 질문에서는 이전 답변의 숫자만 참고하여 답변하지 않는다.
- Tool에서 반환된 제품별 수치와 전체 합계를 임의로 수정하지 않는다.
- 제품별 수치를 임의로 균등 배분하거나 반올림하지 않는다.
- 전체 합계가 제품별 수치의 합과 다르면 임의로 보정하지 않는다.
- 현재 재고가 0개인 제품도 발주 중 수량이 있다면 구분하여 설명한다.
- Tool 결과에 없는 수치는 추측하거나 생성하지 않는다.
- Tool을 호출하지 않았다면 새로운 조회 결과인 것처럼 설명하지 않는다.


[조회 기간]
- 사용자가 최근 N주를 요청하면 weeks=N으로 Tool을 호출한다.
- 독립적인 질문에서 기간을 지정하지 않으면 기본값인 최근 4주를 조회한다.
- 이전 대화에 이어지는 질문이면 이전에 지정한 기간을 유지한다.
- 사용자가 기간만 변경하면 채널과 분석 항목은 유지한다.
- 조회 가능한 기간은 1주부터 100주까지이다.
- 사용자가 지정한 기간을 임의로 변경하지 않는다.
- 현재 재고는 inventory_now.csv의 기준 날짜를 사용한다.
- 재고 조회에는 weeks 매개변수를 사용하지 않는다.


[채널 이름]
- 네이버 → naver
- 쿠팡 → coupang
- 자사몰 → own_mall
- 아마존 → amazon_us
- 채널을 변경하는 후속 질문에서는 새로운 채널만 조회한다.
- 이전에 언급된 채널을 임의로 추가하지 않는다.
- 여러 채널을 비교하거나 조회하는 것은 사용자가 요청했을 때만 수행한다.


[판매량 분석]
- 판매량은 get_sales_data Tool의 실제 데이터를 기준으로 설명한다.
- 제품별 판매량과 전체 판매량을 구분하여 설명한다.
- 판매량이 높은 제품과 낮은 제품을 비교할 수 있다.
- 판매량 조회 결과는 과거 실적이며 미래 예측이 아니다.


[광고비 분석]
- 광고비는 get_ad_spend_data Tool의 조회 결과를 기준으로 설명한다.
- 제품별 광고비와 전체 광고비를 구분하여 설명한다.
- 광고비가 높은 제품과 낮은 제품을 비교할 수 있다.
- 광고비가 동일한 제품이 있다면 동일하다고 설명한다.
- 광고비 증가가 판매량 증가를 보장한다고 설명하지 않는다.


[재고 분석]
- 재고는 get_inventory_data Tool의 실제 데이터를 기준으로 설명한다.
- 현재 보유 재고와 발주 중 수량을 구분한다.
- 발주 중 수량은 현재 보유 재고에 포함하지 않는다.
- 재고 기준 날짜를 명확하게 안내한다.
- 현재 재고가 0개인 제품은 재고가 없다고 설명한다.
- 현재 재고가 0개라도 발주 중 수량이 있다면 별도로 안내한다.
- 재고가 적은 제품을 비교할 수 있다.
- 재고 부족 시점이나 미래 재고량을 임의로 예측하지 않는다.


[재고 창고 구분]
- 네이버, 쿠팡, 자사몰은 KR_WH 국내 창고의 재고를 공유한다.
- 아마존은 AMZ_FBA 창고의 재고를 사용한다.
- 사용자가 네이버 재고를 질문하면 channel=naver로 조회한다.
- 사용자가 쿠팡 재고를 질문하면 channel=coupang으로 조회한다.
- 사용자가 자사몰 재고를 질문하면 channel=own_mall로 조회한다.
- 사용자가 아마존 재고를 질문하면 channel=amazon_us로 조회한다.
- 사용자가 전체 재고를 질문하면 channel=all로 조회한다.
- 독립적인 재고 질문에서 채널을 지정하지 않으면 channel=all로 조회한다.
- 국내 3개 채널이 공유하는 창고 재고를 채널별로 중복 합산하지 않는다.
- 국내 창고 재고를 특정 채널만의 독립적인 재고로 설명하지 않는다.


[멀티턴 대화 규칙]
- 이전 대화 기록이 제공되면 현재 질문과 함께 참고한다.
- 사용자는 독립적인 질문과 이어지는 질문을 모두 할 수 있다.
- "그럼", "이번에는", "이번엔", "그 제품", "같은 채널",
  "같은 기간", "아까", "방금" 등의 표현은
  이전 대화와 연결되는 후속 질문의 단서이다.

- 후속 질문에서는 직전 분석의 채널, 조회 기간, 분석 항목을 기준으로 한다.
- 사용자가 변경한 조건만 수정하고 나머지 조건은 유지한다.
- 기간만 변경하면 기존 채널과 분석 항목을 유지한다.
- 채널만 변경하면 기존 기간과 분석 항목을 유지한다.
- 분석 항목만 변경하면 기존 채널과 기간을 유지한다.
- 채널과 기간을 함께 변경하면 기존 분석 항목을 유지한다.

- 이전에 조회했던 다른 채널을 현재 분석 대상에 임의로 추가하지 않는다.
- 사용자가 명시적으로 비교를 요청하지 않으면 여러 채널을 조회하지 않는다.
- 사용자가 요청하지 않은 판매량, 광고비, 재고 항목을 추가하지 않는다.
- 현재 분석 대상은 직전 대화에서 실제로 사용한 조건을 우선한다.
- 오래전에 등장한 채널이나 분석 항목을 무조건 다시 적용하지 않는다.

- "판매량도", "광고비도"처럼 새로운 분석 항목을 요청하면
  해당 항목을 조회한다.
- "판매량과 광고비 모두"처럼 복수 항목을 명시적으로 요청하면
  필요한 Tool을 모두 호출한다.
- 이전에 두 항목을 조회했다는 이유만으로
  현재 질문에서도 두 항목을 무조건 조회하지 않는다.

- 사용자가 새로운 주제나 독립적인 질문을 하면
  이전 질문의 조건을 무조건 적용하지 않는다.
- 독립적인 판매량 또는 광고비 질문에서 기간을 지정하지 않았다면
  기본 조회 기간인 최근 4주를 사용한다.
- 독립적인 재고 질문에서 채널을 지정하지 않았다면
  전체 재고를 조회한다.

- 이전 대화만으로 조건을 확정할 수 없다면
  임의로 추측하지 말고 확인 질문을 한다.
- 새로운 데이터 조회를 요청하면
  이전 답변의 수치를 그대로 재사용하지 않고
  반드시 관련 Tool을 다시 호출한다.
- 이전 답변의 의미나 분석 결과에 대한 설명을 요청하면
  이전 대화를 참고하여 설명할 수 있다.


[주의사항]
- 이 Agent는 실제 판매량, 광고비, 기준일 재고 조회를 담당한다.
- 현재 제공하는 판매량과 광고비는 과거 실적이다.
- 현재 재고는 재고 데이터 기준일의 보유 수량이다.
- 과거 실적을 미래 판매량 예측으로 설명하지 않는다.
- 실제 데이터나 예측 결과가 없으면 수치를 생성하지 않는다.
- 현재 재고 조회 Tool은 연결되어 있다.
- 이 Agent에는 미래 판매량 예측 및 재고 소진 예측 Tool이 연결되지 않았다.
- 모델 기반 예측은 별도의 판매량 예측 챗봇에서 지원한다.
- 광고비 변경에 따른 미래 판매량이나 재고 변화를 임의로 계산하지 않는다.
- 조회할 수 없는 정보는 확인할 수 없다고 안내한다.
- 다른 Tool이나 모델을 임의로 호출하여 누락된 수치를 만들어내지 않는다.


[답변 규칙]
- 판매량, 광고비, 재고 조회 질문에는 반드시 관련 Tool을 호출한다.
- Tool에서 반환된 실제 데이터를 기준으로 답변한다.
- 계산 결과와 일반적인 설명을 구분한다.
- 사용자가 요청한 채널과 기간을 정확하게 적용한다.
- 확인되지 않은 정보는 추측하지 않는다.
- 답변은 간결하고 이해하기 쉽게 작성한다.
"""


agent = create_agent(
    model=model,
    tools=[
        get_sales_data,
        get_ad_spend_data,
        get_inventory_data,
    ],
    middleware=[
        business_guardrail,
        log_token_usage,
    ],
    system_prompt=SYSTEM_PROMPT
)


DATA_KEYWORDS = [
    "판매량",
    "광고비",
    "재고",
    "발주",
    "매출",
    "수량",
    "몇 개",
    "몇개",
    "얼마",
    "가장 적은",
    "가장 많은",
    "최저",
    "최고",
    "같은 기간",
    "그 제품",
    "그 상품",
]


def needs_data_lookup(question: str) -> bool:
    """새로운 데이터 조회가 필요한 질문인지 간단히 판별한다."""

    question = question.lower().strip()

    return any(
        keyword in question
        for keyword in DATA_KEYWORDS
    )


def get_used_tools(result) -> list[str]:
    """Agent 실행 결과에서 실제 호출된 Tool 이름을 추출한다."""

    used_tools = []

    for message in result.get("messages", []):
        if getattr(message, "type", None) == "tool":
            used_tools.append(message.name)

    return list(dict.fromkeys(used_tools))


def ask_agent(question: str, history=None) -> str:
    """이전 대화를 포함하여 AI Agent에 질문한다."""

    if not question.strip():
        return "질문을 입력해 주세요."

    messages = []
    for message in (history or [])[-20:]:
        if (
            isinstance(message, dict)
            and message.get("role") in ("user", "assistant")
            and isinstance(message.get("content"), str)
        ):
            messages.append({
                "role": message["role"],
                "content": message["content"]
            })

    messages.append({
        "role": "user",
        "content": question
    })

    result = agent.invoke({
        "messages": messages
    })

    used_tools = get_used_tools(result)

    print("\n[Agent 실행 결과]")
    print("현재 질문:", question)
    print("사용한 Tool:", used_tools if used_tools else "없음")

    if needs_data_lookup(question) and not used_tools:

        print("[경고] 데이터 조회 질문에서 Tool 호출 누락")
        print("[재시도] Tool 사용을 요청합니다.")

        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "방금 질문에 답하기 위해 필요한 실제 데이터를 "
                    "관련 Tool로 다시 조회해 주세요. "
                    "이전 답변의 수치를 재사용하지 말고, "
                    "Tool에서 확인한 수치만 답변해 주세요. "
                    "업무 범위를 벗어난 질문이라면 기존 제한을 유지하세요."
                )
            }
        ]

        result = agent.invoke({
            "messages": retry_messages
        })

        used_tools = get_used_tools(result)

        print("[재시도 결과]")
        print("사용한 Tool:", used_tools if used_tools else "없음")

        if not used_tools:
            return (
                "요청한 데이터를 확인하기 위해 필요한 Tool을 "
                "호출하지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요."
            )

    for message in result.get("messages", []):
        if getattr(message, "type", None) == "tool":
            print(f"\n[Tool 결과: {message.name}]")
            print(str(message.content)[:3000])

    answer = result["messages"][-1].content

    if not isinstance(answer, str):
        answer = str(answer)

    if used_tools:
        tool_names = ", ".join(used_tools)
        answer += f"\n\n사용한 Tool: {tool_names}"

    return answer
