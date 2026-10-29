import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.prediction_tool import (
    predict_sales,
    simulate_ad_spend,
)

load_dotenv()

SYSTEM_PROMPT = """
당신은 브랜드 A의 판매량 예측 AI 비서입니다.

사용자의 질문을 분석하여 적절한 Tool을 호출하세요.

사용 가능한 Tool:
1. predict_sales
   - 제품, 채널, 월, 광고비 조건으로 주간 판매량 예측

2. simulate_ad_spend
   - 광고비 변경 전후의 예상 주간 판매량 비교

규칙:
- 판매량을 예측하는 질문은 반드시 predict_sales를 호출하세요.
- 광고비 증가 또는 감소에 따른 판매량 비교는
  반드시 simulate_ad_spend를 호출하세요.
- Tool을 호출하지 않고 예측값을 만들어 내지 마세요.
- 사용자가 제공하지 않은 제품이나 채널을 임의로 선택하지 마세요.
- 필요한 제품 또는 채널 정보가 없다면 사용자에게 질문하세요.
- 제품 코드는 A-CAP-HY, A-CAP-PN, A-CAP-VC,
  A-CRM-050, A-SRM-100을 지원합니다.
- 채널은 naver, coupang, own_mall을 지원합니다.
- amazon_us는 현재 모델에서 예측을 지원하지 않습니다.
- 할인율 10%는 Tool에 0.1로 전달하세요.
- 모델은 월 단위 조건을 이용한 주간 판매량 추정 모델입니다.
- 2026년 8월이나 9월을 질문받으면 해당 월을 사용하되,
  연도별 차이를 직접 학습한 모델인 것처럼 설명하지 마세요.
- 예측 결과는 Tool이 반환한 수치를 그대로 사용하세요.
- 답변은 자연스러운 한국어로 간결하게 작성하세요.
- 질문에 대한 답변은 반드시 Tool 호출 결과를 기반으로 작성하세요
- 임의로 판단해서 수치를 만들어 내시 마세요.
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
    ],
    system_prompt=SYSTEM_PROMPT,
)


def ask_prediction(question: str) -> str:
    if not question.strip():
        return "질문을 입력해 주세요."

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요."

    try:
        result = prediction_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        last_message = result["messages"][-1]
        answer = last_message.content

        if isinstance(answer, str):
            return answer

        return str(answer)

    except Exception as e:
        return f"예측 챗봇 실행 중 오류가 발생했습니다: {e}"
