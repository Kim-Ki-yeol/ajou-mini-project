
import streamlit as st
import pandas as pd
from pathlib import Path
from src.agent import ask_agent


# 화면 설정
st.set_page_config(
    page_title="AI 재고 코파일럿",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path(__file__).parent / "data"

st.markdown("""
<style>
    /* 사이드바 너비 */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
    }

    /* 사이드바 내부 여백 */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }

    /* 사이드바 제목 크기 */
    section[data-testid="stSidebar"] h1 {
        font-size: 28px !important;
        font-weight: 700 !important;
        white-space: nowrap;
    }

    /* 메뉴 글씨 크기 */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 19px !important;
        font-weight: 500 !important;
    }

    /* 메뉴 사이 간격 */
    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 12px;
    }

    /* 메뉴 항목 디자인 */
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 10px 12px;
        border-radius: 10px;
        transition: background-color 0.2s;
    }

    /* 마우스를 올렸을 때 */
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: rgba(128, 128, 128, 0.12);
    }

    /* 선택된 메뉴 배경 */
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: rgba(255, 75, 75, 0.12);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sales():
    return pd.read_csv(
        DATA_DIR / "sales_weekly.csv",
        encoding="utf-8-sig",
        parse_dates=["week_start"]
    )

sales = load_sales()

with st.sidebar:
    st.title("AI 재고 코파일럿")
    st.divider()

    menu = st.radio(
        "메뉴 선택",
        [
            "판매량 분석",
            "광고비 분석",
            "재고 분석",
            "판매량 예측",
            "AI Agent"
        ],
        label_visibility="collapsed",
        key="sidebar_menu"
    )

st.title("AI 재고 코파일럿")
st.caption("판매량 · 광고비 · 재고 분석 및 AI 기반 의사결정 지원")
st.divider()

if menu == "판매량 분석":
    st.subheader("주별 판매량 분석")

    col1, col2 = st.columns(2)

    with col1:
        channel = st.selectbox(
            "판매 채널 선택",
            ["전체"] + sorted(sales["channel"].unique().tolist())
        )

    with col2:
        product = st.selectbox(
            "제품 선택",
            ["전체"] + sorted(sales["product_name"].unique().tolist())
        )

    filtered = sales.copy()

    if channel != "전체":
        filtered = filtered[filtered["channel"] == channel]

    if product != "전체":
        filtered = filtered[filtered["product_name"] == product]

    weekly = (
        filtered.groupby("week_start", as_index=False)["units"]
        .sum()
        .sort_values("week_start")
    )

    st.line_chart(
        weekly,
        x="week_start",
        y="units",
        x_label="주차",
        y_label="판매량 (개)"
    )

elif menu == "광고비 분석":
    st.subheader("광고비 분석")
    st.write("광고비와 판매량의 관계를 분석합니다.")

    ad_channel = st.selectbox(
        "광고비를 분석할 채널 선택",
        sorted(sales["channel"].unique().tolist()),
        key="ad_channel"
    )

    ad_data = sales[sales["channel"] == ad_channel].copy()

    weekly_ad = (
        ad_data.groupby("week_start", as_index=False)
        .agg(
            ad_spend=("ad_spend", "first"),
            units=("units", "sum")
        )
        .sort_values("week_start")
    )

    st.divider()

    st.subheader("광고비와 판매량 관계")

    st.scatter_chart(
        weekly_ad,
        x="ad_spend",
        y="units",
        x_label="광고비",
        y_label="판매량 (개)"
    )

    st.caption(
        "점 하나는 1주일의 데이터입니다. "
        "오른쪽 위로 점들이 모이면 광고비와 판매량이 함께 증가하는 경향이 있다는 뜻입니다."
    )

    st.divider()

    st.subheader("주별 광고비 추이")

    st.line_chart(
        weekly_ad,
        x="week_start",
        y="ad_spend",
        x_label="주차",
        y_label="광고비"
    )

    with st.expander("광고비 분석 데이터 확인"):
        st.dataframe(
            weekly_ad,
            use_container_width=True,
            hide_index=True
        )

elif menu == "재고 분석":
    st.subheader("재고 분석")
    st.write("창고별 제품 재고 현황을 확인합니다.")

    try:
        inventory = pd.read_csv(
            DATA_DIR / "inventory_now.csv",
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        inventory = pd.read_csv(
            DATA_DIR / "inventory_now.csv",
            encoding="cp949"
        )

    warehouse = st.selectbox(
        "창고 선택",
        ["전체"] + sorted(inventory["location"].unique().tolist()),
        key="warehouse"
    )

    stock_data = inventory.copy()

    if warehouse != "전체":
        stock_data = stock_data[
            stock_data["location"] == warehouse
        ]

    total_stock = stock_data["on_hand"].sum()
    incoming_stock = stock_data["open_po_qty"].sum()
    product_count = stock_data["sku"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("현재 보유 재고", f"{total_stock:,.0f}개")

    with col2:
        st.metric("입고 예정 수량", f"{incoming_stock:,.0f}개")

    with col3:
        st.metric("관리 제품 수", f"{product_count}개")

    st.divider()

    st.subheader("제품별 현재 재고")

    product_stock = (
        stock_data.groupby("product_name", as_index=False)["on_hand"]
        .sum()
        .sort_values("on_hand", ascending=False)
    )

    st.bar_chart(
        product_stock,
        x="product_name",
        y="on_hand",
        x_label="제품명",
        y_label="재고 수량 (개)"
    )

    st.divider()

    st.subheader("현재 재고 vs 입고 예정 수량")

    stock_compare = (
        stock_data.groupby("product_name", as_index=False)
        .agg(
            현재_재고=("on_hand", "sum"),
            입고_예정=("open_po_qty", "sum")
        )
    )

    st.bar_chart(
        stock_compare,
        x="product_name",
        y=["현재_재고", "입고_예정"]
    )

    st.caption(
        "입고 예정 수량은 아직 사용 가능한 재고가 아닙니다."
    )

    st.divider()

    st.subheader("③ 재고 상세 현황")

    stock_table = stock_data[
        [
            "location",
            "product_name",
            "on_hand",
            "open_po_qty",
            "lead_time_days",
            "moq"
        ]
    ].rename(
        columns={
            "location": "창고",
            "product_name": "제품명",
            "on_hand": "현재 재고",
            "open_po_qty": "입고 예정 수량",
            "lead_time_days": "리드타임(일)",
            "moq": "최소 주문 수량"
        }
    )

    st.dataframe(
        stock_table,
        use_container_width=True,
        hide_index=True
    )


elif menu == "판매량 예측":
    st.subheader("판매량 예측")
    st.info("모델이 완성되면 Tool을 구성하고 연동할 예정입니다.")

elif menu == "AI Agent":
    st.subheader("AI 재고 코파일럿")
    st.caption("광고비, 판매량, 재고에 대해 질문해 보세요.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("궁금한 점을 입력 하세요 ")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("AI가 분석 중입니다..."):
                try:
                    # 방금 추가한 현재 질문을 제외한 이전 대화를 함께 전달합니다.
                    answer = ask_agent(
                        question,
                        history=st.session_state.messages[:-1],
                    )
                    st.markdown(answer)

                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
                    st.error(answer)

        # AI 응답 기록 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
