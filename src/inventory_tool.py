
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
            기본값 all. 재고 판매량은 항상 국내 3개 채널 합산.
    """

    try:
        sku = sku.strip().upper()
        channel = channel.strip().lower()

        if sku not in SKU_MAP:
            raise ValueError(f"지원하지 않는 제품: {sku}")

        if channel not in CHANNEL_NAMES:
            raise ValueError(f"지원하지 않는 채널: {channel}")

        if not math.isfinite(change_percent):
            raise ValueError(
                "광고비 변경률은 유한한 숫자여야 합니다."
            )

        if change_percent < -100:
            raise ValueError(
                "광고비 감소율은 -100% 이상이어야 합니다."
            )

        if month is not None and not 1 <= month <= 12:
            raise ValueError("예측 월은 1~12 사이여야 합니다.")

        # 국내 창고 재고 불러오기
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

        if not all(math.isfinite(v) for v in (
            current_stock, open_po, lead_days
        )):
            raise ValueError("재고 데이터에 유효하지 않은 숫자가 있습니다.")

        if current_stock < 0 or open_po < 0:
            raise ValueError(
                "재고 및 발주 수량을 확인하세요."
            )

        if lead_days < 0 or moq <= 0:
            raise ValueError(
                "리드타임 또는 최소 발주 수량이 잘못되었습니다."
            )

        # 국내 3개 채널 판매량 예측
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
            original_units = float(run_prediction(conditions))

            changed_conditions = conditions.copy()

            # 선택한 채널만 광고비 변경
            if (
                channel == "all"
                or channel == current_channel
            ):
                changed_conditions["ad_spend"] = (
                    original_ad
                    * (1 + change_percent / 100)
                )

            changed_units = float(
                run_prediction(changed_conditions)
            )

            if not all(math.isfinite(v) for v in (
                original_units, changed_units
            )):
                raise ValueError(
                    "예측 모델이 유효하지 않은 판매량을 반환했습니다."
                )

            original_total += original_units
            changed_total += changed_units

            channel_name = CHANNEL_NAMES[current_channel]

            channel_results.append(
                f"- {channel_name}: "
                f"광고비 {original_ad:,.0f}원 → "
                f"{changed_conditions['ad_spend']:,.0f}원, "
                f"주간 판매량 {original_units:.2f}개 → "
                f"{changed_units:.2f}개"
            )

        if changed_total <= 0:
            return (
                f"제품: {sku}\n"
                "예상 주간 판매량이 0개 이하이므로 "
                "재고 소진 기간을 계산할 수 없습니다."
            )

        # 재고 소진 기간
        stock_weeks = current_stock / changed_total
        stock_days = stock_weeks * 7

        # 리드타임 동안 예상 판매량
        lead_weeks = lead_days / 7

        expected_sales_during_lead = (
            changed_total * lead_weeks
        )

        stock_at_lead = (
            current_stock - expected_sales_during_lead
        )

        # 4주 목표 재고 기준
        target_weeks = 4
        target_stock = changed_total * target_weeks

        # 현재 재고만 기준으로 부족한 수량
        stock_shortage = max(
            0.0,
            target_stock - current_stock,
        )

        # 기존 발주 물량까지 고려한 추가 필요 수량
        additional_needed = max(
            0.0,
            stock_shortage - open_po,
        )

        # 최소 발주 수량(MOQ) 단위로 올림
        suggested_order = (
            math.ceil(additional_needed / moq) * moq
            if additional_needed > 0
            else 0
        )

        # 품절 위험과 발주 판단을 별도로 계산
        if stock_at_lead < 0:
            stockout_risk = "높음"
        else:
            stockout_risk = "낮음"

        if suggested_order > 0:
            order_decision = (
                "4주 목표 재고 기준 추가 발주 검토가 필요합니다."
            )
        elif stock_shortage > 0:
            order_decision = (
                "현재 보유 재고는 4주 목표보다 부족하지만, "
                "발주 진행 수량을 고려하면 "
                "추가 발주 검토 수량은 0개입니다. "
                "기존 발주 물량의 입고 일정을 확인해야 합니다."
            )
        else:
            order_decision = (
                "현재 보유 재고가 4주 목표를 충족하여 "
                "추가 발주가 필요하지 않습니다."
            )

        if stock_at_lead < 0:
            order_decision += (
                " 단, 리드타임 내 품절이 예상되므로 "
                "긴급 재고 확보가 필요합니다."
            )

        # 제품명 및 데이터 기준일
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
            f"[재고 시뮬레이션]\n"
            f"제품: {product_name}\n"
            f"SKU: {sku}\n"
            f"재고 기준일: {as_of}\n"
            f"광고비 변경 대상: {CHANNEL_NAMES[channel]}\n"
            f"판매량 합산 대상: 네이버, 쿠팡, 자사몰\n"
            f"광고비 변경률: {change_percent:+.1f}%\n\n"

            f"[채널별 예측 결과]\n"
            f"{channel_text}\n\n"

            f"[판매량 및 재고]\n"
            f"기존 예상 주간 총판매량: {original_total:.2f}개\n"
            f"변경 후 예상 주간 총판매량: {changed_total:.2f}개\n"
            f"현재 보유 재고: {current_stock:,.0f}개\n"
            f"발주 진행 수량: {open_po:,.0f}개\n"
            f"예상 재고 소진 기간: {stock_weeks:.2f}주\n"
            f"예상 재고 소진 일수: {stock_days:.1f}일\n\n"

            f"[품절 위험]\n"
            f"리드타임: {lead_days:.0f}일\n"
            f"리드타임 종료 시 예상 재고: "
            f"{stock_at_lead:.2f}개\n"
            f"리드타임 내 품절 위험: {stockout_risk}\n\n"

            f"[추가 발주 판단]\n"
            f"4주 목표 재고: {target_stock:.2f}개\n"
            f"현재 재고 기준 부족 수량: "
            f"{stock_shortage:.2f}개\n"
            f"기존 발주 진행 수량: {open_po:,.0f}개\n"
            f"추가 발주 검토 수량(MOQ 반영): "
            f"{suggested_order:,.0f}개\n"
            f"발주 판단: {order_decision}\n\n"

            "참고: 지정한 채널의 광고비만 변경합니다. "
            "국내 전체 채널을 선택하면 3개 채널의 광고비를 "
            "모두 동일 비율로 변경합니다. "
            "재고 소진 기간은 국내 3개 채널의 예상 판매량을 "
            "합산하여 계산했습니다. "
            "주간 판매량이 일정하다고 가정하며, "
            "기존 발주 물량의 실제 입고일은 반영하지 않았습니다."
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        OverflowError,
    ) as e:
        return f"재고 시뮬레이션 실패: {e}"
