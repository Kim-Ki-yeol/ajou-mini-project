from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv(
    Path(__file__).resolve().parent.parent / ".env",
    override=True
)

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.5
)

SYSTEM_PROMPT = """
너는 화장품 브랜드의 재고 관리 AI 코파일럿이야.

- 판매량, 광고비, 재고, 할인율 관련 질문에만 답변한다.
- 누구나 이해하기 쉽게 한국어로 설명한다.
- 확인되지 않은 수치나 정보를 임의로 만들지 않는다.
- 데이터 조회 기능이 없다면 조회할 수 없다고 안내한다.
"""

agent = create_agent(
    model=model,
    tools=[],
    system_prompt=SYSTEM_PROMPT
)

def ask_agent(question):
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    return result["messages"][-1].content