
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool

from src.forecast import predict_units


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "sales_weekly.csv"
)

SKU_MAP = {
    "A-CAP-HY": 0,
    "A-CAP-PN": 1,
    "A-CAP-VC": 2,
    "A-CRM-050": 3,
    "A-SRM-100": 4,
}

CHANNEL_MAP = {
    "coupang": (0, 0),
    "naver": (1, 0),
    "own_mall": (0, 1),
}

FEATURE_COLUMNS = [
    "week_start",
    "sku",
    "ad_spend",
    "promo_days",
    "promo_discount",
    "channel_naver",
    "channel_own_mall",
]


def get_conditions(sku, channel, month=None,
                   ad_spend=None, promo_days=None,
                   promo_discount=None):

    sku = sku.strip().upper()
    channel = channel.strip().lower()

    if channel == "amazon_us":
        raise ValueError(
            "아마존 미국은 현재 모델의 학습 대상에서 제외되어 "
            "판매량 예측을 지원하지 않습니다."
        )

    if sku not in SKU_MAP:
        raise ValueError(f"지원하지 않는 제품: {sku}")

    if channel not in CHANNEL_MAP:
        raise ValueError(f"지원하지 않는 채널: {channel}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["week_start"] = pd.to_datetime(df["week_start"])

    filtered = df[
        (df["sku"] == sku) &
        (df["channel"] == channel)
    ].sort_values("week_start")

    if filtered.empty:
        raise ValueError("해당 제품과 채널의 데이터가 없습니다.")

    latest = filtered.iloc[-1]

    selected_month = (
        int(month) if month is not None
        else int(latest["week_start"].month)
    )
    selected_ad = (
        float(ad_spend) if ad_spend is not None
        else float(latest["ad_spend"])
    )
    selected_promo_days = (
        int(promo_days) if promo_days is not None
        else int(latest["promo_days"])
    )
    selected_discount = (
        float(promo_discount)
        if promo_discount is not None
        else float(latest["promo_discount"])
    )

    if not 1 <= selected_month <= 12:
        raise ValueError("월은 1~12 사이여야 합니다.")
    if selected_ad < 0:
        raise ValueError("광고비는 0원 이상이어야 합니다.")
    if not 0 <= selected_promo_days <= 7:
        raise ValueError("프로모션 일수는 0~7일이어야 합니다.")
    if not 0 <= selected_discount <= 1:
        raise ValueError("할인율은 0~1 사이의 소수로 입력하세요.")

    naver, own_mall = CHANNEL_MAP[channel]

    return {
        "week_start": selected_month,
        "sku": SKU_MAP[sku],
        "ad_spend": selected_ad,
        "promo_days": selected_promo_days,
        "promo_discount": selected_discount,
        "channel_naver": naver,
        "channel_own_mall": own_mall,
    }


def run_prediction(conditions):
    features = pd.DataFrame([conditions])[FEATURE_COLUMNS]
    prediction = float(predict_units(features)[0])
    return max(0.0, prediction)


@tool
def predict_sales(
    sku: str,
    channel: str,
    month: int = None,
    ad_spend: float = None,
    promo_days: int = None,
    promo_discount: float = None,
) -> str:
    """제품과 채널의 판매량을 예측합니다.

    Args:
        sku: 제품 코드. 예: A-CAP-HY
        channel: coupang, naver, own_mall
        month: 예측 월(1~12). 생략하면 최근 기록의 월
        ad_spend: 광고비(원). 생략하면 최근 기록 기준
        promo_days: 주간 프로모션 일수(0~7)
        promo_discount: 할인율. 10%는 0.1
    """

    try:
        conditions = get_conditions(
            sku, channel, month, ad_spend,
            promo_days, promo_discount
        )

        prediction = run_prediction(conditions)

        return (
            f"제품: {sku.upper()}\n"
            f"채널: {channel.lower()}\n"
            f"예측 월: {conditions['week_start']}월\n"
            f"광고비: {conditions['ad_spend']:,.0f}원\n"
            f"프로모션: {conditions['promo_days']}일\n"
            f"할인율: {conditions['promo_discount']:.1%}\n"
            f"예상 주간 판매량: {prediction:.2f}개"
        )

    except (ValueError, FileNotFoundError) as e:
        return f"예측 실패: {e}"


@tool
def simulate_ad_spend(
    sku: str,
    channel: str,
    change_percent: float,
    month: int = None,
) -> str:
    """광고비 변경 전후의 예상 주간 판매량을 비교합니다.

    Args:
        sku: 제품 코드. 예: A-CAP-HY
        channel: coupang, naver, own_mall
        change_percent: 광고비 증감률(%). 예: 20, -10
        month: 예측 월(1~12). 생략하면 최근 기록의 월
    """

    try:
        if change_percent < -100:
            raise ValueError(
                "광고비 감소율은 -100% 이상이어야 합니다."
            )

        conditions = get_conditions(
            sku, channel, month=month
        )

        original_ad = conditions["ad_spend"]
        original_units = run_prediction(conditions)

        changed = conditions.copy()
        changed["ad_spend"] = (
            original_ad * (1 + change_percent / 100)
        )

        changed_units = run_prediction(changed)
        difference = changed_units - original_units

        return (
            f"제품: {sku.upper()}\n"
            f"채널: {channel.lower()}\n"
            f"예측 월: {conditions['week_start']}월\n"
            f"광고비 변경률: {change_percent:+.1f}%\n"
            f"기존 광고비: {original_ad:,.0f}원\n"
            f"변경 광고비: {changed['ad_spend']:,.0f}원\n"
            f"기존 예상 판매량: {original_units:.2f}개\n"
            f"변경 예상 판매량: {changed_units:.2f}개\n"
            f"예상 판매량 차이: {difference:+.2f}개"
        )

    except (ValueError, FileNotFoundError) as e:
        return f"시뮬레이션 실패: {e}"
