from pathlib import Path
import math

import pandas as pd
from langchain_core.tools import tool
from src.forecast import predict_units

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_weekly.csv"
SKU_MAP = {
    "A-CAP-HY": 0, "A-CAP-PN": 1, "A-CAP-VC": 2,
    "A-CRM-050": 3, "A-SRM-100": 4,
}
SKU_NAMES = {
    "A-CAP-HY": "수분 캡슐 앰플", "A-CAP-PN": "진정 캡슐 앰플",
    "A-CAP-VC": "광채 캡슐 앰플", "A-CRM-050": "장벽 크림",
    "A-SRM-100": "발효 퍼스트 앰플",
}
CHANNEL_MAP = {"coupang": (0, 0), "naver": (1, 0), "own_mall": (0, 1)}
CHANNEL_NAMES = {"coupang": "쿠팡", "naver": "네이버", "own_mall": "자사몰"}
FEATURE_COLUMNS = [
    "week_start", "sku", "ad_spend", "promo_days", "promo_discount",
    "channel_naver", "channel_own_mall",
]


def validate_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}은 숫자여야 합니다.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}은 숫자여야 합니다.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}은 유효한 숫자여야 합니다.")
    return number


def validate_weeks(weeks):
    value = validate_number(weeks, "예측 기간")
    if not value.is_integer() or not 1 <= value <= 12:
        raise ValueError("예측 기간은 1~12주 사이의 정수여야 합니다.")
    return int(value)


def normalize_sku(sku):
    if not isinstance(sku, str):
        raise ValueError("제품 코드를 입력해 주세요.")
    value = sku.strip().upper().replace(" ", "")
    if value in {"ALL", "전체", "전체제품", "모든제품"}:
        return "all"
    for code, name in SKU_NAMES.items():
        if value == name.replace(" ", ""):
            return code
    if value not in SKU_MAP:
        raise ValueError(f"지원하지 않는 제품: {sku}")
    return value


def normalize_channel(channel):
    if not isinstance(channel, str):
        raise ValueError("채널을 입력해 주세요.")
    value = channel.strip().lower().replace(" ", "")
    aliases = {
        "전체": "all", "전체채널": "all", "모든채널": "all", "국내전체": "all",
        "쿠팡": "coupang", "네이버": "naver", "자사몰": "own_mall",
        "공식몰": "own_mall", "아마존": "amazon_us", "아마존미국": "amazon_us",
    }
    value = aliases.get(value, value)
    if value == "amazon_us":
        raise ValueError("아마존 미국은 현재 판매량 예측을 지원하지 않습니다.")
    if value != "all" and value not in CHANNEL_MAP:
        raise ValueError(f"지원하지 않는 채널: {channel}")
    return value


def select_targets(sku, channel):
    sku, channel = normalize_sku(sku), normalize_channel(channel)
    skus = list(SKU_MAP) if sku == "all" else [sku]
    channels = list(CHANNEL_MAP) if channel == "all" else [channel]
    return [(s, c) for s in skus for c in channels]


def load_sales_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = {"week_start", "sku", "channel", "ad_spend", "promo_days", "promo_discount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 필수 컬럼 누락: {', '.join(sorted(missing))}")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    if df["week_start"].isna().any():
        raise ValueError("CSV 날짜 형식이 올바르지 않습니다.")
    return df


def get_conditions(sku, channel, month=None, ad_spend=None,
                   promo_days=None, promo_discount=None, df=None):
    sku, channel = normalize_sku(sku), normalize_channel(channel)
    if sku == "all" or channel == "all":
        raise ValueError("get_conditions에는 개별 제품과 채널을 입력해야 합니다.")
    if df is None:
        df = load_sales_data()
    matched = df.loc[(df["sku"] == sku) & (df["channel"] == channel)]
    if matched.empty:
        raise ValueError(f"해당 제품/채널의 데이터가 없습니다: {sku} / {channel}")
    latest = matched.sort_values("week_start").iloc[-1]

    selected_month = validate_number(
        latest["week_start"].month if month is None else month, "월"
    )
    selected_ad = validate_number(
        latest["ad_spend"] if ad_spend is None else ad_spend, "광고비"
    )
    selected_days = validate_number(
        latest["promo_days"] if promo_days is None else promo_days, "프로모션 일수"
    )
    selected_discount = validate_number(
        latest["promo_discount"] if promo_discount is None else promo_discount, "할인율"
    )
    if not selected_month.is_integer() or not 1 <= selected_month <= 12:
        raise ValueError("월은 1~12 사이의 정수여야 합니다.")
    if selected_ad < 0:
        raise ValueError("광고비는 0원 이상이어야 합니다.")
    if not selected_days.is_integer() or not 0 <= selected_days <= 7:
        raise ValueError("프로모션 일수는 0~7 사이의 정수여야 합니다.")
    if not 0 <= selected_discount <= 1:
        raise ValueError("할인율은 0~1 사이여야 합니다.")
    naver, own_mall = CHANNEL_MAP[channel]
    return {
        "week_start": int(selected_month), "sku": SKU_MAP[sku],
        "ad_spend": selected_ad, "promo_days": int(selected_days),
        "promo_discount": selected_discount,
        "channel_naver": naver, "channel_own_mall": own_mall,
    }


def run_prediction(conditions):
    features = pd.DataFrame([conditions])[FEATURE_COLUMNS]
    prediction = float(predict_units(features)[0])
    if not math.isfinite(prediction):
        raise ValueError("모델이 유효하지 않은 예측값을 반환했습니다.")
    return max(0.0, prediction)


def run_period_prediction(conditions, weeks):
    weeks = validate_weeks(weeks)
    weekly = run_prediction(conditions)
    return weekly, weekly * weeks


def calculate_predictions(sku, channel, weeks=1, month=None, ad_spend=None,
                          promo_days=None, promo_discount=None, change_percent=None):
    weeks = validate_weeks(weeks)
    targets = select_targets(sku, channel)
    df = load_sales_data()
    if change_percent is not None:
        change_percent = validate_number(change_percent, "광고비 변경률")
        if change_percent < -100:
            raise ValueError("광고비 감소율은 -100% 이상이어야 합니다.")
    results = []
    for selected_sku, selected_channel in targets:
        conditions = get_conditions(
            selected_sku, selected_channel, month, ad_spend,
            promo_days, promo_discount, df=df,
        )
        weekly, total = run_period_prediction(conditions, weeks)
        item = {
            "sku": selected_sku, "channel": selected_channel,
            "month": conditions["week_start"],
            "ad": conditions["ad_spend"], "weekly": weekly, "total": total,
        }
        if change_percent is not None:
            changed = conditions.copy()
            changed["ad_spend"] *= 1 + change_percent / 100
            if not math.isfinite(changed["ad_spend"]):
                raise ValueError("변경 광고비가 유효 범위를 벗어났습니다.")
            changed_weekly, changed_total = run_period_prediction(changed, weeks)
            item.update(changed_ad=changed["ad_spend"],
                        changed_weekly=changed_weekly, changed_total=changed_total)
        results.append(item)
    return results


def target_label(sku, channel):
    sku, channel = normalize_sku(sku), normalize_channel(channel)
    product = "전체 제품" if sku == "all" else f"{SKU_NAMES[sku]} ({sku})"
    market = "국내 전체 채널" if channel == "all" else CHANNEL_NAMES[channel]
    return f"{product} / {market}"


@tool
def predict_sales(sku: str, channel: str, month: int = None,
                  ad_spend: float = None, promo_days: int = None,
                  promo_discount: float = None, weeks: int = 1) -> str:
    """판매량 예측. sku는 제품 코드/제품명 또는 all(전체 제품), channel은
    coupang/naver/own_mall 또는 all(국내 전체 채널). 월과 주간 광고비를
    생략하면 각 조합의 최근 CSV 기록을 사용합니다. weeks는 1~12주이며
    1개월=4주, 2개월=8주, 3개월=12주로 계산합니다.
    """
    try:
        weeks = validate_weeks(weeks)
        results = calculate_predictions(
            sku, channel, weeks, month, ad_spend, promo_days, promo_discount
        )
        total = sum(x["total"] for x in results)
        weekly = sum(x["weekly"] for x in results)
        lines = [
            "[판매량 예측]", f"대상: {target_label(sku, channel)}",
            f"기간: {weeks}주 / 조합: {len(results)}개",
            f"주간 광고비 합계: {sum(x['ad'] for x in results):,.0f}원",
            f"주간 예상 판매량: {weekly:.2f}개",
            f"{weeks}주 예상 판매량: {total:.2f}개", "", "[상세]",
        ]
        for x in results:
            lines.append(
                f"- {SKU_NAMES[x['sku']]} / {CHANNEL_NAMES[x['channel']]} "
                f"({x['month']}월): 주간 광고비 {x['ad']:,.0f}원, "
                f"주간 {x['weekly']:.2f}개, {weeks}주 {x['total']:.2f}개"
            )
        lines.append("※ 같은 월·광고비·프로모션 조건을 매주 적용한 시나리오입니다. ")
        lines.append("※ 미래 주차별 변화를 별도로 예측한 결과는 아닙니다.")
        return "\n".join(lines)
    except (ValueError, FileNotFoundError) as exc:
        return f"예측 실패: {exc}"


@tool
def simulate_ad_spend(sku: str, channel: str, change_percent: float,
                      month: int = None, weeks: int = 1) -> str:
    """광고비 증감률(%)에 따른 판매량 비교. change_percent는 20이면
    20% 증가, -10이면 10% 감소. sku는 제품 코드/제품명 또는 all,
    channel은 coupang/naver/own_mall 또는 all. 광고비 금액을 묻지 말고
    각 제품·채널의 최근 CSV 광고비를 기준으로 계산합니다. weeks는 1~12주.
    """
    try:
        weeks = validate_weeks(weeks)
        results = calculate_predictions(
            sku, channel, weeks=weeks, month=month, change_percent=change_percent
        )
        old_total = sum(x["total"] for x in results)
        new_total = sum(x["changed_total"] for x in results)
        difference = new_total - old_total
        rate = f"{difference / old_total * 100:+.2f}%" if old_total > 0 else "계산 불가"
        lines = [
            "[광고비 시뮬레이션]", f"대상: {target_label(sku, channel)}",
            f"기간: {weeks}주 / 조합: {len(results)}개",
            f"광고비 변경률: {float(change_percent):+.1f}%",
            f"주간 광고비 합계: {sum(x['ad'] for x in results):,.0f}원 → "
            f"{sum(x['changed_ad'] for x in results):,.0f}원",
            f"주간 예상 판매량: {sum(x['weekly'] for x in results):.2f}개 → "
            f"{sum(x['changed_weekly'] for x in results):.2f}개",
            f"{weeks}주 예상 판매량: {old_total:.2f}개 → {new_total:.2f}개",
            f"판매량 차이: {difference:+.2f}개 / 변화율: {rate}", "", "[상세]",
        ]
        for x in results:
            lines.append(
                f"- {SKU_NAMES[x['sku']]} / {CHANNEL_NAMES[x['channel']]} "
                f"({x['month']}월): 광고비 {x['ad']:,.0f}원 → "
                f"{x['changed_ad']:,.0f}원, "
                f"{weeks}주 판매량 {x['total']:.2f}개 → {x['changed_total']:.2f}개"
            )
        lines.append("※ 모든 선택 조합에 같은 광고비 증감률을 적용했습니다.")
        lines.append("※ 동일 조건의 주간 예측을 기간만큼 합산한 모델 추정치입니다.")
        return "\n".join(lines)
    except (ValueError, FileNotFoundError) as exc:
        return f"시뮬레이션 실패: {exc}"


@tool
def simulate_discount(
    sku: str,
    channel: str,
    change_percent: float,
    month: int = None,
    weeks: int = 4,
) -> str:
    """할인율 변경에 따른 예상 판매량 비교.
    change_percent=10은 할인율을 10%p 증가, -10은 10%p 감소시킵니다.
    각 제품·채널의 최근 CSV 할인율을 기준으로 계산합니다.
    sku와 channel은 all을 지원하며 weeks는 1~12주입니다.
    """
    try:
        weeks = validate_weeks(weeks)
        change_percent = validate_number(change_percent, "할인율 변경 폭")
        targets = select_targets(sku, channel)
        df = load_sales_data()
        results = []

        for selected_sku, selected_channel in targets:
            original = get_conditions(
                selected_sku,
                selected_channel,
                month=month,
                df=df,
            )
            old_discount = original["promo_discount"]
            new_discount = old_discount + change_percent / 100

            if not 0 <= new_discount <= 1:
                raise ValueError(
                    f"{SKU_NAMES[selected_sku]} / "
                    f"{CHANNEL_NAMES[selected_channel]}의 변경 할인율 "
                    f"{new_discount * 100:.1f}%가 0~100% 범위를 벗어났습니다."
                )

            changed = original.copy()
            changed["promo_discount"] = new_discount
            old_weekly, old_total = run_period_prediction(original, weeks)
            new_weekly, new_total = run_period_prediction(changed, weeks)
            results.append({
                "sku": selected_sku,
                "channel": selected_channel,
                "old_discount": old_discount,
                "new_discount": new_discount,
                "old_weekly": old_weekly,
                "new_weekly": new_weekly,
                "old_total": old_total,
                "new_total": new_total,
            })

        old_total = sum(x["old_total"] for x in results)
        new_total = sum(x["new_total"] for x in results)
        difference = new_total - old_total
        rate = (
            f"{difference / old_total * 100:+.2f}%"
            if old_total > 0 else "계산 불가"
        )
        lines = [
            "[할인율 시뮬레이션]",
            f"대상: {target_label(sku, channel)}",
            f"기간: {weeks}주 / 조합: {len(results)}개",
            f"할인율 변경: {change_percent:+.1f}%p",
            f"주간 예상 판매량: "
            f"{sum(x['old_weekly'] for x in results):.2f}개 → "
            f"{sum(x['new_weekly'] for x in results):.2f}개",
            f"{weeks}주 예상 판매량: {old_total:.2f}개 → {new_total:.2f}개",
            f"판매량 차이: {difference:+.2f}개",
            f"변화율: {rate}",
            "",
            "[제품별 상세]",
        ]
        for x in results:
            lines.append(
                f"- {SKU_NAMES[x['sku']]} / {CHANNEL_NAMES[x['channel']]}: "
                f"할인율 {x['old_discount'] * 100:.1f}% → "
                f"{x['new_discount'] * 100:.1f}%, "
                f"{weeks}주 예상 판매량 "
                f"{x['old_total']:.2f}개 → {x['new_total']:.2f}개"
            )
        lines.append("※ 광고비와 프로모션 일수는 기존 조건을 유지했습니다.")
        lines.append("※ 동일한 주간 조건을 반복 적용한 모델 추정치입니다.")
        lines.append("※ 실제 할인 효과나 매출·이익 증가를 보장하지 않습니다.")
        return "\n".join(lines)
    except (ValueError, FileNotFoundError) as exc:
        return f"할인율 시뮬레이션 실패: {exc}"
