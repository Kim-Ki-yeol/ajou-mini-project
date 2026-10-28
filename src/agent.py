
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.tool import calculate_ad_budget
from src.middleware import (
    business_guardrail,
    log_token_usage,
)


# 환경변수 로드
load_dotenv(
    Path(__file__).resolve().parent.parent / ".env",
    override=True
)


# LLM 설정
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


SYSTEM_PROMPT = """
너는 브랜드 A의 재고 관리 AI 코파일럿이야.

[역할]
- 온라인 판매 채널의 광고비, 판매량, 재고 관리를 지원한다.
- 모든 답변은 한국어로 작성한다.

[Tool 사용]
- 광고비 변경 금액을 계산할 때 calculate_ad_budget Tool을 사용한다.
- 실제 판매량 예측 모델은 아직 연결되지 않았다.
- 실제 데이터나 예측 결과가 없으면 임의로 수치를 생성하지 않는다.

[답변 규칙]
- 계산 결과와 일반적인 설명을 구분한다.
- 확인되지 않은 정보는 추측하지 않는다.
"""


agent = create_agent(
    model=model,
    tools=[calculate_ad_budget],
    middleware=[
        business_guardrail,
        log_token_usage,
    ],
    system_prompt=SYSTEM_PROMPT
)


def ask_agent(question: str) -> str:

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    used_tools = []

    for message in result["messages"]:
        if getattr(message, "type", None) == "tool":
            used_tools.append(message.name)

    answer = result["messages"][-1].content

    if used_tools:
        tool_names = ", ".join(dict.fromkeys(used_tools))
        answer += f"\n\n사용한 Tool: {tool_names}"

    return answer
