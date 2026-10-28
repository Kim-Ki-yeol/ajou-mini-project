
from langchain.tools import tool


@tool
def calculate_ad_budget(
    budget: int,
    change_percent: float
) -> str:
    """
    광고비 변경 후 예산을 계산합니다.

    Args:
        budget: 현재 광고비 (원)
        change_percent: 광고비 변경 비율 (%)
    """

    if budget < 0:
        return "광고비는 0원 이상이어야 합니다."

    if change_percent < -100:
        return "광고비는 100%를 초과하여 감소시킬 수 없습니다."

    new_budget = budget * (1 + change_percent / 100)

    return (
        f"기존 광고비: {budget:,}원\n"
        f"변경 비율: {change_percent:+g}%\n"
        f"변경 후 광고비: {new_budget:,.0f}원"
    )
