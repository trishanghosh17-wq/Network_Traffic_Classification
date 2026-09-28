"""Shared model runtime for Streamlit and FastAPI.

Both services load the exact same canonical artifact and schema from models/.
This prevents silent model/feature-version drift between the UI and API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
META_PATH = ROOT / "models" / "model_metadata.json"
OOD_PATH = ROOT / "models" / "ood_isolation_forest_demo.joblib"
OOD_META_PATH = ROOT / "models" / "ood_metadata.json"
STATS_PATH = ROOT / "app" / "feature_stats.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def artifact_sha256(path: Path = MODEL_PATH) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRuntime:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Canonical model missing: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        self.schema = _load_json(SCHEMA_PATH)
        self.metadata = _load_json(META_PATH)
        self.features = list(self.schema["features"])
        self.stats = _load_json(STATS_PATH) if STATS_PATH.exists() else {}
        self.booster = self.model.named_steps["model"].get_booster()
        self.imputer = self.model[:-1]
        self.model_sha256 = artifact_sha256()
        expected = self.metadata.get("artifact_sha256")
        if expected and expected != self.model_sha256:
            raise RuntimeError("Model checksum mismatch: artifact and metadata are inconsistent.")
        self.ood_metadata = _load_json(OOD_META_PATH) if OOD_META_PATH.exists() else {}
        ood_features = self.ood_metadata.get('features', self.features)
        self.ood = joblib.load(OOD_PATH) if OOD_PATH.exists() and ood_features == self.features else None

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return frame[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def predict(self, frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
        X = self.prepare(frame)
        prob = self.model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        contrib = self.booster.predict(xgb.DMatrix(self.imputer.transform(X)), pred_contribs=True)[:, :-1]
        ood_flag = np.zeros(len(X), dtype=bool)
        ood_score = np.full(len(X), np.nan)
        if self.ood is not None:
            filled = X.fillna(0.0)
            ood_score = self.ood.decision_function(filled)
            cutoff = float(self.ood_metadata.get("cutoff_decision_function", -0.0))
            ood_flag = ood_score < cutoff
        confidence = np.maximum(prob, 1 - prob)
        review = ood_flag | (confidence < 0.65)
        return {"probability": prob, "prediction": pred, "contrib": contrib,
                "ood_flag": ood_flag, "ood_score": ood_score, "confidence": confidence,
                "review_flag": review, "prepared": X}

    def top_factors(self, contrib_row: np.ndarray, n: int = 8) -> list[dict[str, float | str]]:
        idx = np.argsort(-np.abs(contrib_row))[:n]
        return [{"feature": self.features[int(i)], "contribution_log_odds": round(float(contrib_row[int(i)]), 6)} for i in idx]

    def range_warnings(self, values: dict[str, float | None]) -> list[str]:
        warnings = []
        for feature, value in values.items():
            if value is None or not np.isfinite(float(value)):
                warnings.append(f"{feature}: value is missing or non-finite; median imputation will be used")
                continue
            stat = self.stats.get(feature)
            if stat and (value < stat.get("min", value) or value > stat.get("max", value)):
                warnings.append(f"{feature}={value:g} is outside the training range ({stat.get('min', 0):g} to {stat.get('max', 0):g})")
        return warnings
