
from pathlib import Path

import pandas as pd
from langchain.tools import tool
from src.forecast import predict_units
from src.forecast import load_reference_data


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




from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
import math

from langchain.tools import tool
from src.forecast import load_reference_data, predict_units


CHANNEL_ENCODING = {
    "coupang": (0, 0),
    "naver": (1, 0),
    "own_mall": (0, 1),
}


@tool
def predict_sales(
    sku: int | None = None,
    week_start: int | None = None,
    ad_spend: float | None = None,
    promo_days: float | None = None,
    promo_discount: float | None = None,
    channel: Literal["coupang", "naver", "own_mall"] | None = None,
) -> dict:
    """기준 조건과 사용자 지정 조건의 예상 판매량을 비교합니다.

    사용자가 명시한 조건만 전달하고 나머지는 생략합니다.
    생략된 조건은 CSV와 현재 날짜에서 자동 계산합니다.

    Args:
        sku: 변경할 SKU 숫자 코드. 제품명으로부터 추측하지 않습니다.
        week_start: 변경할 월(1~12). 날짜나 주차 번호가 아닙니다.
        ad_spend: 변경 후 광고비. 학습 데이터와 동일한 단위입니다.
        promo_days: 변경 후 주간 프로모션 일수(0~7).
        promo_discount: 변경 후 할인율. 학습 데이터와 동일한 단위입니다.
        channel: 변경 후 채널(coupang, naver, own_mall).

    기준 조건:
        SKU는 전체 최빈값, 월은 한국 시간 기준 현재 월,
        광고비와 할인율은 기준 SKU의 평균,
        프로모션 일수는 전체 평균, 채널은 쿠팡입니다.
    """
    try:
        df = load_reference_data()

        modes = df["sku"].mode()
        if modes.empty:
            raise ValueError("기준 SKU를 선택할 데이터가 없습니다.")

        baseline_sku = int(modes.iloc[0])
        sku_df = df[df["sku"] == baseline_sku]
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        baseline = {
            "week_start": now.month,
            "sku": baseline_sku,
            "ad_spend": float(sku_df["ad_spend"].mean()),
            "promo_days": float(df["promo_days"].mean()),
            "promo_discount": float(sku_df["promo_discount"].mean()),
            "channel_naver": 0,
            "channel_own_mall": 0,
        }

        requested = {
            "sku": sku,
            "week_start": week_start,
            "ad_spend": ad_spend,
            "promo_days": promo_days,
            "promo_discount": promo_discount,
        }

        # None만 제외하므로 광고비 0, 할인율 0도 정상 적용됩니다.
        supplied = {
            key: value
            for key, value in requested.items()
            if value is not None
        }

        for key, value in supplied.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{key}는 유한한 숫자여야 합니다.")

        if sku is not None and sku not in df["sku"].unique():
            raise ValueError("CSV에 없는 SKU 코드입니다.")

        if week_start is not None and not 1 <= week_start <= 12:
            raise ValueError("week_start는 월을 뜻하며 1~12입니다.")

        if ad_spend is not None and ad_spend < 0:
            raise ValueError("광고비는 0 이상이어야 합니다.")

        if promo_days is not None and not 0 <= promo_days <= 7:
            raise ValueError("프로모션 일수는 0~7이어야 합니다.")

        if promo_discount is not None and promo_discount < 0:
            raise ValueError("할인율은 0 이상이어야 합니다.")

        scenario = baseline.copy()
        scenario.update(supplied)

        if channel is not None:
            naver, own_mall = CHANNEL_ENCODING[channel]
            scenario["channel_naver"] = naver
            scenario["channel_own_mall"] = own_mall

        changed_features = {
            key: {
                "before": baseline[key],
                "after": scenario[key],
            }
            for key in baseline
            if baseline[key] != scenario[key]
        }

        before = predict_units(baseline)

        # 조건이 같으면 같은 예측 결과를 재사용합니다.
        after = (
            predict_units(scenario)
            if changed_features
            else before
        )

        before_units = max(0.0, before["raw_prediction"])
        after_units = max(0.0, after["raw_prediction"])
        difference = after_units - before_units

        warnings = list(dict.fromkeys(
            before["warnings"] + after["warnings"]
        ))

        if len(modes) > 1:
            warnings.append(
                "최빈 SKU가 여러 개여서 첫 번째 코드를 기준으로 선택했습니다."
            )

        if sku is not None and sku != baseline_sku:
            warnings.append(
                "SKU 변경 효과를 비교하기 위해 광고비와 할인율의 "
                "기본값은 기준 SKU의 평균으로 유지했습니다."
            )

        return {
            "status": "ok",
            "result_type": "model_scenario_comparison",
            "reference_date": now.date().isoformat(),
            "baseline_inputs": baseline,
            "scenario_inputs": scenario,
            "baseline_channel": "coupang",
            "scenario_channel": channel or "coupang",
            "changed_features": changed_features,
            "before_predicted_units": round(before_units, 2),
            "after_predicted_units": round(after_units, 2),
            "units_difference": round(difference, 2),
            "units_change_pct": (
                round(difference / before_units * 100, 2)
                if before_units != 0
                else None
            ),
            "warnings": warnings,
        }

    except (ValueError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "message": str(exc),
        }