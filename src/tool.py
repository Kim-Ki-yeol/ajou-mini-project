
from pathlib import Path

import pandas as pd
from langchain.tools import tool


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sales_weekly.csv"
)


@tool
def get_sales_data(channel: str, weeks: int = 4) -> str:
    """
    판매 채널의 최근 N주 제품별 판매량을 조회합니다.
    Args:
        channel: 판매 채널 (own_mall, coupang, naver, amazon_us)
        weeks: 조회할 최근 주 수 (기본값 4주)
    """
    channels = ["own_mall", "coupang", "naver", "amazon_us"]

    if channel not in channels:
        return f"지원하지 않는 채널입니다. 가능한 채널: {', '.join(channels)}"
    
    if weeks < 1 or weeks > 100:
        return "조회 기간은 1~100주 사이여야 합니다."
    
    df = pd.read_csv(DATA_PATH)

    df["week_start"] = pd.to_datetime(df["week_start"])
    channel_df = df[df["channel"] == channel].copy()

    if channel_df.empty:
        return "해당 채널의 판매 데이터가 없습니다."
    latest_week = channel_df["week_start"].max()
    start_week = latest_week - pd.Timedelta(weeks=weeks - 1)
    recent_df = channel_df[
        (channel_df["week_start"] >= start_week)
        & (channel_df["week_start"] <= latest_week)
    ]

    result = (
        recent_df.groupby("product_name")["units"].sum().sort_values(ascending=False)
    )

    if result.empty:
        return "해당 기간에 판매 데이터가 없습니다."
    
    lines = [
        f"판매 채널: {channel}",
        f"조회 기간: {start_week.date()} ~ {latest_week.date()}",
        f"전체 판매량: {result.sum():,}개",
        "",
        "[제품별 판매량]",
    ]

    for product, units in result.items():
        lines.append(f"- {product}: {units:,}개")

    return "\n".join(lines)



@tool
def get_ad_spend_data(channel: str, weeks: int = 4) -> str:
    """
    판매 채널의 최근 N주 제품별 광고비를 조회합니다.

    Args:
        channel: 판매 채널 (own_mall, coupang, naver, amazon_us)
        weeks: 조회할 최근 주 수 (기본값 4주)
    """

    channels = ["own_mall", "coupang", "naver", "amazon_us"]

    if channel not in channels:
        return f"지원하지 않는 채널입니다. 가능한 채널: {', '.join(channels)}"

    if weeks < 1 or weeks > 100:
        return "조회 기간은 1~100주 사이여야 합니다."

    df = pd.read_csv(DATA_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])

    channel_df = df[df["channel"] == channel].copy()

    if channel_df.empty:
        return "해당 채널의 광고비 데이터가 없습니다."

    latest_week = channel_df["week_start"].max()
    start_week = latest_week - pd.Timedelta(weeks=weeks - 1)

    recent_df = channel_df[
        (channel_df["week_start"] >= start_week)
        & (channel_df["week_start"] <= latest_week)
    ]

    if recent_df.empty:
        return "해당 기간에 광고비 데이터가 없습니다."

    currencies = recent_df["currency"].dropna().unique()

    if len(currencies) != 1:
        return "통화 단위를 확인할 수 없거나 혼합되어 있습니다."

    currency = currencies[0]

    # 제품별 광고비 합산
    result = (
        recent_df.groupby("product_name")["ad_spend"]
        .sum()
        .sort_values(ascending=False)
    )

    lines = [
        f"판매 채널: {channel}",
        f"조회 기간: {start_week.date()} ~ {latest_week.date()}",
        f"전체 광고비: {result.sum():,.0f} {currency}",
        "",
        "[제품별 광고비]",
    ]

    for product, spend in result.items():
        lines.append(f"- {product}: {spend:,.0f} {currency}")

    return "\n".join(lines)



@tool
def get_inventory_data(channel: str = "all") -> str:
    """
    판매 채널에서 사용하는 제품별 현재 재고를 조회합니다.

    Args:
        channel: 판매 채널 (all, own_mall, coupang, naver, amazon_us)
    """

    channels = ["all", "own_mall", "coupang", "naver", "amazon_us"]

    if channel not in channels:
        return f"지원하지 않는 채널입니다. 가능한 채널: {', '.join(channels)}"

    inventory_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "inventory_now.csv"
    )

    df = pd.read_csv(inventory_path)

    if df.empty:
        return "재고 데이터가 없습니다."

    if channel == "amazon_us":
        df = df[df["location"] == "AMZ_FBA"].copy()

    elif channel in ["own_mall", "coupang", "naver"]:
        df = df[df["location"] == "KR_WH"].copy()

    if df.empty:
        return "해당 채널에서 사용하는 재고 데이터가 없습니다."

    as_of = df["as_of"].max()

    result = (
        df.groupby("product_name", as_index=False)
        .agg(
            on_hand=("on_hand", "sum"),
            open_po_qty=("open_po_qty", "sum"),
        )
        .sort_values("on_hand", ascending=True)
    )

    lines = [
        f"조회 채널: {channel}",
        f"재고 기준일: {as_of}",
        f"전체 현재 재고: {result['on_hand'].sum():,}개",
        f"전체 발주 중 수량: {result['open_po_qty'].sum():,}개",
        "",
        "[제품별 재고 현황]",
    ]

    for _, row in result.iterrows():
        lines.append(
            f"- {row['product_name']}: "
            f"현재 재고 {row['on_hand']:,}개 / "
            f"발주 중 {row['open_po_qty']:,}개"
        )

    lines.extend([
        "",
        "[참고]",
        "- 국내 채널(네이버, 쿠팡, 자사몰)은 KR_WH 재고만 공유합니다.",
        "- 아마존은 AMZ_FBA 재고만 사용합니다.",
        "- 발주 중 수량은 현재 재고에 포함하지 않습니다.",
    ])

    return "\n".join(lines)
