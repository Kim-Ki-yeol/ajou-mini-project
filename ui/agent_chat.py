import streamlit as st

from src.agent import ask_agent


def render_agent_chat():
    st.subheader("AI 재고 코파일럿")
    st.caption("광고비, 판매량, 재고에 대해 질문해 보세요.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "궁금한 점을 입력하세요.",
        key="agent_chat_input"
    )

    if question:
        history = st.session_state.messages.copy()

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("AI가 분석 중입니다..."):
                try:
                    answer = ask_agent(
                        question,
                        history=history
                    )

                    if not isinstance(answer, str):
                        answer = str(answer)

                    st.markdown(answer)

                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
