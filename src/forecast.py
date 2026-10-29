from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent.parent

# 반드시 학습 코드의 X.columns 순서와 대조하세요.
FEATURE_COLUMNS = [
    "week_start",
    "sku",
    "ad_spend",
    "promo_days",
    "promo_discount",
    "channel_naver",
    "channel_own_mall",
]

# 컬럼 순서, 인코딩, 전처리를 확인한 뒤 True로 변경합니다.
FEATURE_CONTRACT_CONFIRMED = False


@lru_cache(maxsize=1)
def load_forecast_model():
    model = xgb.Booster()
    model.load_model(str(ROOT / "models" / "xgbrfr_model.json"))

    if model.num_features() != len(FEATURE_COLUMNS):
        raise ValueError("모델 입력 특성 수가 예상과 다릅니다.")

    return model


@lru_cache(maxsize=1)
def load_reference_data():
    df = pd.read_csv(ROOT / "data" / "sales_export.csv")
    missing = set(FEATURE_COLUMNS + ["units"]) - set(df.columns)

    if missing:
        raise ValueError(f"CSV 필수 컬럼 누락: {sorted(missing)}")

    return df


def predict_units(features: dict) -> dict:
    if not FEATURE_CONTRACT_CONFIRMED:
        raise ValueError(
            "학습 당시 입력 컬럼 순서와 인코딩 확인이 필요합니다."
        )

    # units는 정답 컬럼이므로 입력에 포함하지 않습니다.
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    X = X.astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("입력값은 유한한 숫자여야 합니다.")

    reference = load_reference_data()
    warnings = []

    for column in FEATURE_COLUMNS:
        value = float(X.iloc[0][column])
        low = float(reference[column].min())
        high = float(reference[column].max())

        if not low <= value <= high:
            warnings.append(
                f"{column}={value}는 참고 CSV 범위 "
                f"({low}~{high}) 밖입니다."
            )

    model = load_forecast_model()
    raw = float(model.predict(xgb.DMatrix(X))[0])

    if not np.isfinite(raw):
        raise ValueError("모델이 유효하지 않은 예측값을 반환했습니다.")

    if raw < 0:
        warnings.append("음수 예측값을 표시 정책에 따라 0으로 보정했습니다.")

    return {
        "status": "ok",
        "result_type": "model_prediction",
        "model_file": "xgbrfr_model.json",
        "inputs": features,
        "raw_prediction": raw,
        "predicted_units": round(max(0.0, raw), 2),
        "warnings": warnings,
    }