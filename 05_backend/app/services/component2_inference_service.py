from functools import lru_cache
from typing import Dict, List
import pandas as pd
from app.ml.component2_rul_predictor import Component2RULPredictor


@lru_cache(maxsize=1)
def get_predictor() -> Component2RULPredictor:
    return Component2RULPredictor()


def predict_batch(df: pd.DataFrame) -> List[Dict]:
    predictor = get_predictor()
    results = []

    if "serial_number" not in df.columns:
        raise ValueError("Missing required column: serial_number")

    if "date" not in df.columns:
        raise ValueError("Missing required column: date")

    base_features = predictor.base_features

    missing_cols = [c for c in base_features if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns: {missing_cols}"
        )

    for sn, g in df.groupby("serial_number"):
        g = g.sort_values("date").copy()
        g_len = len(g)

        if g_len < 30:
            results.append(
                {
                    "serial_number": str(sn),
                    "status": "insufficient_data",
                    "required": 30,
                    "available": g_len,
                }
            )
            continue

        try:
            rul = predictor.predict(g)
            results.append(
                {
                    "serial_number": str(sn),
                    "status": "ok",
                    "predicted_rul_days": round(float(rul), 3),
                    "observations_used": 30,
                }
            )
        except Exception as e:
            results.append(
                {
                    "serial_number": str(sn),
                    "status": "error",
                    "error": str(e),
                }
            )

    return results
