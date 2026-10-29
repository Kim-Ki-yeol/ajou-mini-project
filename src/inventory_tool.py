
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


@tool
def simulate_inventory(
    sku: str,
    change_percent: float,
    month: int = None,
) -> str:
    """광고비 변경에 따른 국내 재고 소진 기간과 발주 필요 여부를 분석합니다.

    Args:
        sku: 제품 코드. 예: A-CAP-HY
        change_percent: 광고비 증감률(%). 예: 50, -20
        month: 예측 월(1~12). 생략하면 최근 판매 기록의 월
    """

    try:
        sku = sku.strip().upper()

        if sku not in SKU_MAP:
            raise ValueError(f"지원하지 않는 제품: {sku}")

        if not math.isfinite(change_percent):
            raise ValueError("광고비 변경률은 유한한 숫자여야 합니다.")

        if change_percent < -100:
            raise ValueError(
                "광고비 감소율은 -100% 이상이어야 합니다."
            )

        inventory = pd.read_csv(
            INVENTORY_PATH,
            encoding="utf-8-sig",
        )

        stock_rows = inventory[
            (inventory["sku"] == sku)
            & (inventory["location"] == "KR_WH")
        ]

        if stock_rows.empty:
            raise ValueError(
                "해당 제품의 국내 창고 재고 정보가 없습니다."
            )

        stock = stock_rows.iloc[0]

        current_stock = float(stock["on_hand"])
        open_po = float(stock["open_po_qty"])
        lead_days = float(stock["lead_time_days"])
        moq = int(stock["moq"])

        if current_stock < 0 or open_po < 0:
            raise ValueError("재고 및 발주 수량을 확인하세요.")

        if lead_days < 0 or moq <= 0:
            raise ValueError("리드타임 또는 최소 발주 수량이 잘못되었습니다.")

        original_total = 0.0
        changed_total = 0.0
        channel_results = []

        for channel in CHANNELS:
            conditions = get_conditions(
                sku=sku,
                channel=channel,
                month=month,
            )

            original_ad = conditions["ad_spend"]
            original_units = run_prediction(conditions)

            changed_conditions = conditions.copy()
            changed_conditions["ad_spend"] = (
                original_ad * (1 + change_percent / 100)
            )

            changed_units = run_prediction(changed_conditions)

            original_total += original_units
            changed_total += changed_units

            channel_results.append(
                f"- {channel}: "
                f"광고비 {original_ad:,.0f}원 → "
                f"{changed_conditions['ad_spend']:,.0f}원, "
                f"판매량 {original_units:.2f}개 → "
                f"{changed_units:.2f}개"
            )

        if changed_total <= 0:
            return (
                f"제품: {sku}\n"
                "예상 주간 판매량이 0개이므로 "
                "재고 소진 기간을 계산할 수 없습니다."
            )

        stock_weeks = current_stock / changed_total
        stock_days = stock_weeks * 7

        lead_weeks = lead_days / 7
        expected_sales_during_lead = (
            changed_total * lead_weeks
        )

        stock_at_lead = (
            current_stock - expected_sales_during_lead
        )

        target_weeks = 4
        target_stock = changed_total * target_weeks

        additional_needed = max(
            0.0,
            target_stock - current_stock - open_po,
        )

        suggested_order = (
            math.ceil(additional_needed / moq) * moq
            if additional_needed > 0
            else 0
        )

        if stock_at_lead < 0:
            decision = (
                "리드타임 내 품절 위험이 있습니다. "
                "긴급 재고 확보가 필요합니다."
            )
        elif suggested_order > 0:
            decision = (
                "리드타임 내 품절 위험은 낮지만, "
                "4주 목표 재고 기준 추가 발주 검토가 필요합니다."
            )
        else:
            decision = (
                "현재 가정에서는 리드타임 내 품절 위험이 낮고, "
                "4주 목표 재고 기준 추가 발주가 필요하지 않습니다."
            )

        product_name = (
            stock["product_name"]
            if "product_name" in inventory.columns
            else sku
        )

        as_of = (
            stock["as_of"]
            if "as_of" in inventory.columns
            else "확인 필요"
        )

        channel_text = "\n".join(channel_results)

        return (
            f"제품: {product_name}\n"
            f"SKU: {sku}\n"
            f"재고 기준일: {as_of}\n"
            f"대상 채널: 네이버, 쿠팡, 자사몰\n"
            f"광고비 변경률: {change_percent:+.1f}%\n\n"
            f"[채널별 예측 결과]\n"
            f"{channel_text}\n\n"
            f"기존 예상 주간 총판매량: {original_total:.2f}개\n"
            f"변경 후 예상 주간 총판매량: {changed_total:.2f}개\n"
            f"현재 재고: {current_stock:,.0f}개\n"
            f"발주 진행 수량: {open_po:,.0f}개\n"
            f"예상 재고 소진 기간: {stock_weeks:.2f}주\n"
            f"예상 재고 소진 일수: {stock_days:.1f}일\n"
            f"리드타임: {lead_days:.0f}일\n"
            f"리드타임 종료 시 예상 재고: {stock_at_lead:.2f}개\n"
            f"4주 목표 기준 추가 발주 검토 수량: "
            f"{suggested_order:,.0f}개\n"
            f"발주 판단: {decision}\n\n"
            "참고: 국내 3개 채널의 광고비를 동일 비율로 변경하고 "
            "주간 판매량이 일정하게 유지된다고 가정했습니다. "
            "기존 발주 물량의 실제 입고일은 반영하지 않았습니다."
        )

    except (
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as e:
        return f"재고 시뮬레이션 실패: {e}"
