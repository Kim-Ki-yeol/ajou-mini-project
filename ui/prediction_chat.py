
import streamlit as st

from src.prediction_chatbot import ask_prediction


def render_prediction_chat():
    st.subheader("AI 판매량 예측 챗봇")
    st.caption(
        "XGBoost 기반 판매량 예측 및 광고비·재고 시뮬레이션"
    )

    if "prediction_messages" not in st.session_state:
        st.session_state.prediction_messages = []

    for message in st.session_state.prediction_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "판매량, 광고비, 재고에 대해 질문해 보세요.",
        key="prediction_chat_input"
    )

    if question:
        history = st.session_state.prediction_messages.copy()

        st.session_state.prediction_messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("AI가 분석 중입니다..."):
                try:
                    answer = ask_prediction(
                        question,
                        history=history
                    )

                    if not isinstance(answer, str):
                        answer = str(answer)

                    st.markdown(answer)

                except Exception as e:
                    answer = f"예측 실행 중 오류가 발생했습니다: {e}"
                    st.error(answer)

        st.session_state.prediction_messages.append({
            "role": "assistant",
            "content": answer
        })
