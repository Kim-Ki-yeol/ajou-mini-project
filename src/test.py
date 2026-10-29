import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("PYTHON:", sys.executable)
print("PROJECT_ROOT:", ROOT)

try:
    from src.forecast import FEATURE_COLUMNS, load_forecast_model

    model = load_forecast_model()

    input_data = {
        "week_start": 10,
        "sku": 0,
        "ad_spend": 789041,
        "promo_days": 1.06,
        "promo_discount": 0.0605,
        "channel_naver": 0,
        "channel_own_mall": 0,
    }

    X_new = pd.DataFrame([input_data], columns=FEATURE_COLUMNS).astype(float)
    prediction = float(model.predict(X_new)[0])
    print(f"예상 판매량: {prediction:.0f}개")
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
    traceback.print_exc()
    raise
