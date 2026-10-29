
from pathlib import Path

import pandas as pd
from langchain.tools import tool
from src.forecast import predict_units


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




@tool
def predict_sales(
    week_start: int,
    sku: int,
    ad_spend: float,
    promo_days: int,
    promo_discount: float,
    channel_naver: int,
    channel_own_mall: int,
) -> dict:
    """학습된 XGBoost 모델로 한 행의 판매량을 예측합니다.

    모든 입력은 학습 당시와 동일한 인코딩·단위를 사용해야 합니다.
    week_start와 sku는 학습 시 변환된 코드입니다.
    채널은 학습 당시의 두 더미 변수로 전달합니다.
    인코딩이나 입력 조건이 확인되지 않으면 호출하지 마세요.
    과거 실적 조회 또는 재고 소진일 계산 도구가 아닙니다.
    """
    if ad_spend < 0:
        return {"status": "error", "message": "광고비는 0 이상이어야 합니다."}

    if not 0 <= promo_days <= 7:
        return {"status": "error", "message": "프로모션 일수는 0~7입니다."}

    if channel_naver not in (0, 1) or channel_own_mall not in (0, 1):
        return {"status": "error", "message": "채널 코드는 0 또는 1입니다."}

    if channel_naver + channel_own_mall > 1:
        return {"status": "error", "message": "두 채널을 동시에 지정할 수 없습니다."}

    features = {
        "week_start": week_start,
        "sku": sku,
        "ad_spend": ad_spend,
        "promo_days": promo_days,
        "promo_discount": promo_discount,
        "channel_naver": channel_naver,
        "channel_own_mall": channel_own_mall,
    }

    try:
        return predict_units(features)
    except (ValueError, FileNotFoundError) as exc:
        return {"status": "error", "message": str(exc)}


from datetime import datetime
from zoneinfo import ZoneInfo
import math

from langchain.tools import tool

from src.forecast import load_reference_data, predict_units


@tool
def simulate_ad_spend(new_ad_spend: float) -> dict:
    """광고비만 변경했을 때 모델의 예상 판매량 변화를 비교합니다.

    new_ad_spend: 변경 후 광고비. 학습 데이터와 동일한 단위.

    나머지 조건은 자동 적용합니다.
    - SKU: 전체 데이터 최빈값
    - 월: 한국 시간 기준 현재 월
    - 기존 광고비: 선택 SKU의 평균
    - 프로모션 일수: 전체 데이터 평균
    - 할인율: 선택 SKU의 평균
    - 채널: 쿠팡
    """
    if not math.isfinite(new_ad_spend) or new_ad_spend < 0:
        return {
            "status": "error",
            "message": "광고비는 0 이상의 유한한 숫자여야 합니다.",
        }

    try:
        df = load_reference_data()

        modes = df["sku"].mode()
        if modes.empty:
            raise ValueError("기준 SKU를 선택할 수 없습니다.")

        # 최빈값이 여러 개라면 mode()가 반환한 첫 번째 값 선택
        sku = int(modes.iloc[0])
        sku_df = df[df["sku"] == sku]

        baseline_ad_spend = float(sku_df["ad_spend"].mean())
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        # 두 예측에서 공통으로 사용할 조건
        common = {
            "week_start": now.month,
            "sku": sku,
            "promo_days": float(df["promo_days"].mean()),
            "promo_discount": float(sku_df["promo_discount"].mean()),
            "channel_naver": 0,
            "channel_own_mall": 0,
        }

        before = predict_units({
            **common,
            "ad_spend": baseline_ad_spend,
        })

        after = predict_units({
            **common,
            "ad_spend": float(new_ad_spend),
        })

        # 반올림 전 값으로 변화량 계산
        before_units = max(0.0, before["raw_prediction"])
        after_units = max(0.0, after["raw_prediction"])
        difference = after_units - before_units

        change_pct = (
            difference / before_units * 100
            if before_units != 0
            else None
        )

        warnings = list(dict.fromkeys(
            before["warnings"] + after["warnings"]
        ))

        if len(modes) > 1:
            warnings.append(
                "최빈 SKU가 여러 개여서 첫 번째 코드를 선택했습니다."
            )

        return {
            "status": "ok",
            "comparison_type": "model_prediction_vs_model_prediction",
            "reference_date": now.date().isoformat(),
            "channel": "coupang",
            "fixed_inputs": common,
            "baseline_ad_spend": baseline_ad_spend,
            "new_ad_spend": float(new_ad_spend),
            "before_predicted_units": round(before_units, 2),
            "after_predicted_units": round(after_units, 2),
            "units_difference": round(difference, 2),
            "units_change_pct": (
                round(change_pct, 2)
                if change_pct is not None
                else None
            ),
            "warnings": warnings,
        }

    except (ValueError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "message": str(exc),
        }