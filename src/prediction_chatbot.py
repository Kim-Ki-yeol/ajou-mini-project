def ask_prediction(question: str) -> str:
    if not question.strip():
        return "질문을 입력해 주세요."

    return "예측 챗봇 연결 성공! 실제 예측 모델은 아직 연결되지 않았습니다."