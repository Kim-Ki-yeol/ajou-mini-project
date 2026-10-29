
from pathlib import Path
import xgboost as xgb
import pandas as pd
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "xgbrfr_model.json"
)

SCALER_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "robust_scaler.pkl"
)

FEATURE_COLUMNS = [
    "week_start",
    "sku",
    "ad_spend",
    "promo_days",
    "promo_discount",
    "channel_naver",
    "channel_own_mall"
]


def load_forecast_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"예측 모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    model = xgb.XGBRFRegressor()
    model.load_model(str(MODEL_PATH))

    return model


def load_forecast_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"스케일러 파일을 찾을 수 없습니다: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    return scaler


def predict_units(features: pd.DataFrame):
    model = load_forecast_model()
    scaler = load_forecast_scaler()

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"모델 입력에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    features = features[FEATURE_COLUMNS]

    expected_count = model.n_features_in_

    if features.shape[1] != expected_count:
        raise ValueError(
            f"모델 입력 특성은 {expected_count}개여야 합니다. "
            f"현재 입력: {features.shape[1]}개"
        )

    if features.isnull().any().any():
        raise ValueError(
            "예측 입력 데이터에 결측값이 있습니다."
        )

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    return prediction
