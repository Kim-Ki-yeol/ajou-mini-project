
from pathlib import Path
import math

import pandas as pd
from langchain_core.tools import tool

from src.prediction_tool import (
    SKU_MAP,
    get_conditions,
    run_prediction,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INVENTORY_PATH = BASE_DIR / "data" / "inventory_now.csv"

CHANNELS = ["naver", "coupang", "own_mall"]

CHANNEL_NAMES = {
    "naver": "네이버",
    "coupang": "쿠팡",
    "own_mall": "자사몰",
    "all": "국내 전체 채널",
}


@tool
def simulate_inventory(
    sku: str,
    change_percent: float = 0,
    month: int = None,
    channel: str = "all",
) -> str:
    """국내 재고 소진 기간, 품절 위험 및 추가 발주 필요 여부를 분석합니다.

    Args:
        sku: 제품 코드. 예: A-CAP-HY
        change_percent: 광고비 증감률(%). 기본값 0
        month: 예측 월(1~12). 생략하면 최근 판매 기록의 월
        channel: 광고비 변경 대상 채널.
            naver, coupang, own_mall, all 중 선택.
    """

    try:
        if not isinstance(sku, str) or not sku.strip():
            raise ValueError("제품 코드를 입력하세요.")

        sku = sku.strip().upper()

        if sku not in SKU_MAP:
            raise ValueError(f"지원하지 않는 제품: {sku}")

        if not isinstance(channel, str):
            raise ValueError("채널은 문자열로 입력하세요.")

        channel = channel.strip().lower()

        if channel not in CHANNEL_NAMES:
            raise ValueError(
                f"지원하지 않는 채널: {channel}. "
                f"가능한 채널: {', '.join(CHANNEL_NAMES)}"
            )

        change_percent = float(change_percent)

        if not math.isfinite(change_percent):
            raise ValueError("광고비 변경률은 유효한 숫자여야 합니다.")

        if change_percent < -100:
            raise ValueError("광고비 감소율은 -100% 이상이어야 합니다.")

        if month is not None:
            month = int(month)
            if not 1 <= month <= 12:
                raise ValueError("예측 월은 1~12 사이여야 합니다.")

        inventory = pd.read_csv(
            INVENTORY_PATH,
            encoding="utf-8-sig",
        )

        required_columns = {
            "sku",
            "location",
            "on_hand",
            "open_po_qty",
            "lead_time_days",
            "moq",
        }

        missing = required_columns - set(inventory.columns)

        if missing:
            raise ValueError(
                f"재고 CSV에 필요한 컬럼이 없습니다: {sorted(missing)}"
            )

        stock_rows = inventory[
            (inventory["sku"].astype(str).str.upper() == sku)
            & (inventory["location"] == "KR_WH")
        ]

        if stock_rows.empty:
            raise ValueError("해당 제품의 국내 창고 재고가 없습니다.")

        stock = stock_rows.iloc[0]

        current_stock = float(stock["on_hand"])
        open_po = float(stock["open_po_qty"])
        lead_days = float(stock["lead_time_days"])
        moq_value = float(stock["moq"])

        values = [current_stock, open_po, lead_days, moq_value]

        if not all(math.isfinite(value) for value in values):
            raise ValueError("재고 CSV에 유효하지 않은 숫자가 있습니다.")

        if current_stock < 0 or open_po < 0:
            raise ValueError("재고 및 발주 수량은 음수일 수 없습니다.")

        if lead_days < 0 or moq_value <= 0:
            raise ValueError("리드타임 또는 MOQ 값이 잘못되었습니다.")

        if not moq_value.is_integer():
            raise ValueError("MOQ는 정수여야 합니다.")

        moq = int(moq_value)

        original_total = 0.0
        changed_total = 0.0
        channel_results = []

        for current_channel in CHANNELS:
            conditions = get_conditions(
                sku=sku,
                channel=current_channel,
                month=month,
            )

            original_ad = float(conditions["ad_spend"])

            if not math.isfinite(original_ad) or original_ad < 0:
                raise ValueError(
                    f"{CHANNEL_NAMES[current_channel]} 광고비 데이터가 잘못되었습니다."
                )

            original_units = float(run_prediction(conditions))

            changed_conditions = conditions.copy()

            if channel == "all" or channel == current_channel:
                changed_conditions["ad_spend"] = (
                    original_ad * (1 + change_percent / 100)
                )

            changed_units = float(run_prediction(changed_conditions))

            if not all(math.isfinite(value) for value in (
                original_units,
                changed_units,
            )):
                raise ValueError("예측 모델이 유효하지 않은 값을 반환했습니다.")

            if original_units < 0 or changed_units < 0:
                raise ValueError("예측 판매량이 음수입니다.")

            original_total += original_units
            changed_total += changed_units

            channel_results.append(
                f"- {CHANNEL_NAMES[current_channel]}: "
                f"광고비 {original_ad:,.0f}원 → "
                f"{changed_conditions['ad_spend']:,.0f}원, "
                f"주간 판매량 {original_units:.2f}개 → "
                f"{changed_units:.2f}개"
            )

        if original_total <= 0 or changed_total <= 0:
            raise ValueError(
                "예상 주간 판매량이 0개 이하이므로 "
                "재고 소진 기간을 계산할 수 없습니다."
            )

        original_weeks = (
            current_stock / original_total
            if original_total > 0
            else None
        )
        changed_weeks = current_stock / changed_total

        original_days = original_weeks * 7
        changed_days = changed_weeks * 7

        weeks_difference = changed_weeks - original_weeks
        days_difference = changed_days - original_days

        lead_weeks = lead_days / 7
        stock_at_lead = current_stock - changed_total * lead_weeks

        target_weeks = 4
        target_stock = changed_total * target_weeks
        stock_shortage = max(0.0, target_stock - current_stock)
        additional_needed = max(0.0, stock_shortage - open_po)

        suggested_order = (
            math.ceil(additional_needed / moq) * moq
            if additional_needed > 0
            else 0
        )

        stockout_risk = (
            "높음"
            if stock_at_lead <= 0
            else "낮음"
        )

        if stockout_risk == "높음":
            order_decision = (
                "리드타임 내 품절 위험이 있습니다. "
                "긴급 재고 확보와 발주 일정을 확인하세요."
            )
        elif suggested_order > 0:
            order_decision = (
                "4주 목표 재고 기준 추가 발주 검토가 필요합니다."
            )
        elif stock_shortage > 0:
            order_decision = (
                "현재 재고는 4주 목표보다 부족하지만, "
                "진행 중인 발주 수량을 고려하면 "
                "추가 발주 검토 수량은 0개입니다. "
                "입고 일정을 확인하세요."
            )
        else:
            order_decision = (
                "현재 재고가 4주 목표를 충족합니다."
            )

        product_name = (
            str(stock["product_name"])
            if "product_name" in inventory.columns
            else sku
        )

        as_of = (
            str(stock["as_of"])
            if "as_of" in inventory.columns
            else "확인 필요"
        )

        return (
            f"[재고 시뮬레이션]\n"
            f"제품: {product_name} ({sku})\n"
            f"재고 기준일: {as_of}\n"
            f"광고비 변경 대상: {CHANNEL_NAMES[channel]}\n"
            f"광고비 변경률: {change_percent:+.1f}%\n"
            f"판매량 합산 대상: 네이버, 쿠팡, 자사몰\n\n"
            f"[채널별 예측 결과]\n"
            f"{chr(10).join(channel_results)}\n\n"
            f"[판매량 및 재고 소진 비교]\n"
            f"변경 전 주간 총판매량: {original_total:.2f}개\n"
            f"변경 후 주간 총판매량: {changed_total:.2f}개\n"
            f"현재 보유 재고: {current_stock:,.0f}개\n"
            f"발주 진행 수량: {open_po:,.0f}개\n"
            f"변경 전 재고 소진 기간: {original_weeks:.2f}주 "
            f"({original_days:.1f}일)\n"
            f"변경 후 재고 소진 기간: {changed_weeks:.2f}주 "
            f"({changed_days:.1f}일)\n"
            f"소진 기간 변화: {weeks_difference:+.2f}주 "
            f"({days_difference:+.1f}일)\n\n"
            f"[품절 위험]\n"
            f"리드타임: {lead_days:.0f}일\n"
            f"리드타임 종료 시 예상 재고: {stock_at_lead:.2f}개\n"
            f"리드타임 내 품절 위험: {stockout_risk}\n\n"
            f"[추가 발주 판단]\n"
            f"4주 목표 재고: {target_stock:.2f}개\n"
            f"현재 재고 기준 부족 수량: {stock_shortage:.2f}개\n"
            f"추가 발주 검토 수량(MOQ 반영): {suggested_order:,.0f}개\n"
            f"발주 판단: {order_decision}\n\n"
            "참고: 재고 소진 기간은 현재 보유 재고만 기준으로 계산했습니다. "
            "발주 진행 수량은 현재 재고에 포함하지 않았습니다. "
            "리드타임 종료 시 예상 재고 계산에는 입고 예정 수량을 반영하지 않았으며, "
            "주간 판매량이 일정하다고 가정한 모델 기반 시뮬레이션입니다."
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        OverflowError,
        ZeroDivisionError,
    ) as error:
        return f"재고 시뮬레이션 실패: {error}"
