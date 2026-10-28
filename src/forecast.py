from pathlib import Path
import xgboost as xgb

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "xgbrfr_model.json"
)


def load_forecast_model():
    model = xgb.XGBRFRegressor()
    model.load_model(str(MODEL_PATH))
    return model