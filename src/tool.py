
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
from pydantic import BaseModel, ConfigDict, Field
from src.forecast import load_reference_data, predict_units


CHANNEL_ENCODING = {
    "coupang": (0, 0),
    "naver": (1, 0),
    "own_mall": (0, 1),
}

# 학습 시 LabelEncoder가 SKU ID 알파벳순으로 부여한 코드입니다.
SKU_ENCODING = {
    "수분 캡슐 앰플 40caps": 0,    # A-CAP-HY
    "진정 캡슐 앰플 40caps": 1,    # A-CAP-PN
    "광채 캡슐 앰플 40caps": 2,    # A-CAP-VC
    "장벽 크림 50ml": 3,           # A-CRM-050
    "발효 퍼스트 앰플 100ml": 4,   # A-SRM-100
}
SKU_NAMES = {code: name for name, code in SKU_ENCODING.items()}


def resolve_sku(product_name: str) -> int:
    """제품명(줄임말 포함)을 SKU 코드로 변환한다."""

    # 입력의 모든 단어가 들어 있는 제품만 후보로 봅니다.
    words = product_name.lower().split()
    matches = [
        code
        for name, code in SKU_ENCODING.items()
        if words and all(word in name.lower() for word in words)
    ]

    if len(matches) == 1:
        return matches[0]

    products = ", ".join(SKU_ENCODING)

    if not matches:
        raise ValueError(
            f"'{product_name}'은(는) 제품 목록에 없습니다. "
            f"예측 가능한 제품: {products}"
        )

    candidates = ", ".join(SKU_NAMES[code] for code in matches)
    raise ValueError(
        f"'{product_name}'에 해당하는 제품이 여러 개입니다: {candidates}"
    )

# SKU 언급이 없을 때 사용하는 기준 SKU입니다.
# 학습 데이터의 SKU별 행 수가 같아 최빈값 대신 0번으로 고정합니다.
BASELINE_SKU = 0
BASELINE_SKU_NAME = SKU_NAMES[BASELINE_SKU]


class PredictSalesInput(BaseModel):
    # 정의되지 않은 인자를 조용히 버리지 않고 오류로 돌려줍니다.
    model_config = ConfigDict(extra="forbid")

    sku: int | None = Field(
        None,
        description=(
            "변경할 SKU 숫자 코드(0~4). 사용자가 코드를 직접 말할 때 사용합니다. "
            "product_name과 함께 전달하지 않습니다."
        ),
    )
    product_name: str | None = Field(
        None,
        description=(
            "변경할 제품명. 사용자가 말한 제품명을 그대로 전달합니다. "
            "목록에 없는 제품이어도 다른 제품으로 바꾸지 않고 그대로 전달합니다. "
            "SKU 코드로 직접 바꾸지 않습니다. "
            f"제품 목록: {', '.join(SKU_ENCODING)}"
        ),
    )
    month: int | None = Field(
        None,
        description=(
            "예측할 월(1~12). '12월'처럼 월을 직접 말할 때 사용합니다. "
            "months_ahead와 함께 전달하지 않습니다."
        ),
    )
    months_ahead: int | None = Field(
        None,
        description=(
            "현재 월로부터 몇 달 뒤인지(0~11). '이번 달'은 0, "
            "'다음 달'은 1, '3개월 뒤'는 3입니다. "
            "상대적인 시점이 언급되면 반드시 전달합니다."
        ),
    )
    ad_spend: float | None = Field(
        None,
        description="변경 후 광고비 절대 금액(원). '100만 원으로 변경'은 1000000입니다.",
    )
    ad_spend_change_pct: float | None = Field(
        None,
        description="기준 광고비 대비 변경률(%). '20% 증가'는 20, '10% 감소'는 -10입니다.",
    )
    ad_spend_change_amount: float | None = Field(
        None,
        description=(
            "기준 광고비 대비 증감 금액(원). '100만 원 증가'는 1000000, "
            "'50만 원 감소'는 -500000입니다. "
            "광고비 인자 3개 중 하나만 전달합니다."
        ),
    )
    promo_days: float | None = Field(
        None,
        description="변경 후 주간 프로모션 일수(0~7).",
    )
    promo_discount: float | None = Field(
        None,
        description="변경 후 할인율. 학습 데이터와 동일한 단위입니다.",
    )
    channel: Literal["coupang", "naver", "own_mall"] | None = Field(
        None,
        description="변경 후 채널(coupang, naver, own_mall).",
    )
    # 기간 누락을 막기 위해 기본값 없이 항상 전달받습니다.
    weeks: int = Field(
        ...,
        description=(
            "필요 수량을 합산할 기간(주, 1~52). 항상 전달합니다. "
            "'4주 동안', '4주차까지', '4주치'는 4이고, "
            "주 단위 기간 언급이 없으면 1입니다. "
            "결과에 1주 예측값 × weeks를 함께 반환합니다."
        ),
    )


@tool(args_schema=PredictSalesInput)
def predict_sales(
    sku: int | None = None,
    product_name: str | None = None,
    month: int | None = None,
    months_ahead: int | None = None,
    ad_spend: float | None = None,
    ad_spend_change_pct: float | None = None,
    ad_spend_change_amount: float | None = None,
    promo_days: float | None = None,
    promo_discount: float | None = None,
    channel: Literal["coupang", "naver", "own_mall"] | None = None,
    weeks: int = 1,
) -> dict:
    """기준 조건과 사용자 지정 조건의 예상 판매량을 비교합니다.

    사용자가 명시한 조건만 전달하고 나머지는 생략합니다.
    생략된 조건은 CSV와 현재 날짜에서 자동 계산합니다.

    기준 조건:
        SKU는 0번(수분 캡슐 앰플 40caps), 월은 한국 시간 기준 현재 월,
        광고비는 기준 SKU의 평균,
        프로모션 일수와 할인율은 기준 SKU의 최빈값, 채널은 쿠팡입니다.
    """
    try:
        df = load_reference_data()

        baseline_sku = BASELINE_SKU
        sku_df = df[df["sku"] == baseline_sku]

        if sku_df.empty:
            raise ValueError("기준 SKU 0번의 데이터가 없습니다.")
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        # 프로모션 조건은 기준 SKU에서 가장 흔한 값을 사용합니다.
        promo_days_modes = sku_df["promo_days"].mode()
        promo_discount_modes = sku_df["promo_discount"].mode()

        baseline = {
            "week_start": now.month,
            "sku": baseline_sku,
            "ad_spend": float(sku_df["ad_spend"].mean()),
            "promo_days": float(promo_days_modes.iloc[0]),
            "promo_discount": float(promo_discount_modes.iloc[0]),
            "channel_naver": 0,
            "channel_own_mall": 0,
        }

        if product_name is not None:
            if sku is not None:
                raise ValueError("sku와 product_name은 함께 지정할 수 없습니다.")

            # 제품명은 학습 때와 같은 SKU 코드로 변환합니다.
            sku = resolve_sku(product_name)

        requested = {
            "sku": sku,
            # 모델 입력 컬럼 week_start는 월을 뜻합니다.
            "week_start": month,
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

        ad_spend_args = [
            value
            for value in (ad_spend, ad_spend_change_pct, ad_spend_change_amount)
            if value is not None
        ]

        if len(ad_spend_args) > 1:
            raise ValueError(
                "ad_spend, ad_spend_change_pct, ad_spend_change_amount 중 "
                "하나만 지정할 수 있습니다."
            )

        for value in ad_spend_args:
            if not math.isfinite(float(value)):
                raise ValueError("광고비 인자는 유한한 숫자여야 합니다.")

        # 상대 변경은 기준 광고비에서 Python이 계산합니다.
        if ad_spend_change_pct is not None:
            supplied["ad_spend"] = (
                baseline["ad_spend"] * (1 + ad_spend_change_pct / 100)
            )

        if ad_spend_change_amount is not None:
            supplied["ad_spend"] = (
                baseline["ad_spend"] + ad_spend_change_amount
            )

        if supplied.get("ad_spend", 0) < 0:
            raise ValueError("변경 후 광고비는 0 이상이어야 합니다.")

        if sku is not None and sku not in df["sku"].unique():
            raise ValueError("CSV에 없는 SKU 코드입니다.")

        if month is not None and not 1 <= month <= 12:
            raise ValueError("month는 1~12입니다.")

        if months_ahead is not None:
            if month is not None:
                raise ValueError(
                    "month와 months_ahead는 함께 지정할 수 없습니다."
                )

            if not 0 <= months_ahead <= 11:
                raise ValueError("months_ahead는 0~11이어야 합니다.")

            supplied["week_start"] = (now.month - 1 + months_ahead) % 12 + 1

        if ad_spend is not None and ad_spend < 0:
            raise ValueError("광고비는 0 이상이어야 합니다.")

        if promo_days is not None and not 0 <= promo_days <= 7:
            raise ValueError("프로모션 일수는 0~7이어야 합니다.")

        if promo_discount is not None and promo_discount < 0:
            raise ValueError("할인율은 0 이상이어야 합니다.")

        if not 1 <= weeks <= 52:
            raise ValueError("weeks는 1~52주여야 합니다.")

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

        if sku is None:
            warnings.append(
                f"SKU를 지정하지 않아 {BASELINE_SKU}번({BASELINE_SKU_NAME})으로 "
                "예측했습니다."
            )

        if len(promo_days_modes) > 1 or len(promo_discount_modes) > 1:
            warnings.append(
                "프로모션 일수 또는 할인율의 최빈값이 여러 개여서 "
                "가장 작은 값을 기준으로 선택했습니다."
            )

        if sku is not None and sku != baseline_sku:
            warnings.append(
                "SKU 변경 효과를 비교하기 위해 광고비(기준 SKU 평균)와 "
                "프로모션 일수·할인율(기준 SKU 최빈값)의 기본값을 유지했습니다."
            )

        if weeks > 1:
            warnings.append(
                f"{weeks}주 합계는 {weeks}주 모두 같은 조건이라고 가정하고 "
                f"1주 예측값에 {weeks}를 곱한 값입니다."
            )

        return {
            "status": "ok",
            "result_type": "model_scenario_comparison",
            "reference_date": now.date().isoformat(),
            "baseline_inputs": baseline,
            "scenario_inputs": scenario,
            "baseline_product_name": SKU_NAMES[baseline["sku"]],
            "scenario_product_name": SKU_NAMES.get(scenario["sku"]),
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
            # n주 필요 수량은 1주 예측값 × n으로 계산합니다.
            "weeks": weeks,
            "before_total_units": round(before_units * weeks, 2),
            "after_total_units": round(after_units * weeks, 2),
            "total_units_difference": round(difference * weeks, 2),
            "warnings": warnings,
        }

    except (ValueError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "message": str(exc),
        }