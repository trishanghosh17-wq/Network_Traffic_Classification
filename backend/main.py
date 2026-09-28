"""Company-oriented DDoS inference API.

Canonical model: models/model.joblib
Shared runtime: common/runtime.py
Run: DDOS_API_KEY=change-me uvicorn backend.main:app --port 8000
"""
from __future__ import annotations

import csv, io, json, logging, os, secrets, sqlite3, threading, time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common.runtime import ModelRuntime, ROOT

API_KEY = os.getenv("DDOS_API_KEY", "")
ALLOW_ANON = os.getenv("DDOS_ALLOW_ANON") == "1"
RUNTIME = ModelRuntime()
with open(ROOT / 'results' / 'final_metrics.json') as _f:
    METRICS = json.load(_f)
THRESHOLD = float(os.getenv("DDOS_THRESHOLD", str(RUNTIME.metadata.get("threshold_default", 0.5))))
HIGH = float(os.getenv("DDOS_HIGH_SEVERITY", "0.9"))
MAX_ROWS = int(os.getenv("DDOS_MAX_BATCH_ROWS", "100000"))
MAX_UPLOAD_MB = int(os.getenv("DDOS_MAX_UPLOAD_MB", "25"))
DB_PATH = os.getenv("DDOS_DB", str(ROOT / "data" / "audit.db"))
MODEL_VERSION = RUNTIME.metadata["model_version"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ddos")
_lock = threading.Lock()

if not API_KEY and not ALLOW_ANON:
    log.warning("DDOS_API_KEY is not set: /api routes require authentication.")

@contextmanager
def db():
    if DB_PATH != ":memory:":
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, source TEXT, label TEXT NOT NULL,
        prob REAL NOT NULL, severity TEXT NOT NULL, model_version TEXT, review_flag INTEGER NOT NULL DEFAULT 0,
        ood_score REAL, factors TEXT, features TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_ts ON predictions(ts)")
    existing = {row[1] for row in c.execute("PRAGMA table_info(predictions)").fetchall()}
    for col, ddl in {
        "review_flag": "ALTER TABLE predictions ADD COLUMN review_flag INTEGER NOT NULL DEFAULT 0",
        "ood_score": "ALTER TABLE predictions ADD COLUMN ood_score REAL",
    }.items():
        if col not in existing:
            c.execute(ddl)

app = FastAPI(title="Network Traffic Detection Service", version="3.1.0", description="DDoS flow classifier with exact tree-SHAP factors and advisory OOD screening.")


def auth(x_api_key: Optional[str] = Header(default=None)):
    if ALLOW_ANON:
        return
    if not API_KEY or not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Invalid or missing API key")

class FlowIn(BaseModel):
    features: dict[str, Optional[float]]
    source: str = Field(default="manual", max_length=64)


def severity(p: float) -> str:
    return "high" if p >= HIGH else "medium" if p >= THRESHOLD else "none"


def record(source, label, p, factors, feats, review_flag, ood_score):
    return (time.time(), source, label, float(p), severity(float(p)), MODEL_VERSION, int(bool(review_flag)),
            None if not np.isfinite(ood_score) else float(ood_score), json.dumps(factors), json.dumps(feats))


def save(rows):
    if not rows:
        return
    with db() as c:
        c.executemany("INSERT INTO predictions(ts,source,label,prob,severity,model_version,review_flag,ood_score,factors,features) VALUES(?,?,?,?,?,?,?,?,?,?)", rows)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "model_checksum": RUNTIME.model_sha256[:16], "canonical_artifact": "models/model.joblib"}

@app.get("/api/model", dependencies=[Depends(auth)])
def model_info():
    return {
        "version": MODEL_VERSION,
        "schema_version": RUNTIME.metadata.get("feature_schema_version"),
        "checksum_sha256": RUNTIME.model_sha256,
        "threshold_default": THRESHOLD,
        "threshold_candidate_internal_holdout": RUNTIME.metadata.get("threshold_candidate_from_internal_holdout"),
        "high_severity": HIGH,
        "classes": RUNTIME.metadata.get("classes"),
        "features": RUNTIME.features,
        "holdout_metrics": METRICS,
        "validation_status": RUNTIME.metadata.get("validation_status"),
        "limitations": RUNTIME.metadata.get("limitations", []),
    }

@app.get("/api/presets", dependencies=[Depends(auth)])
def presets():
    with open(ROOT / "results" / "scenario_presets.json") as f:
        return json.load(f)

@app.post("/api/predict", dependencies=[Depends(auth)])
def predict(flow: FlowIn):
    missing = [f for f in RUNTIME.features if f not in flow.features]
    if missing:
        raise HTTPException(422, {"message": "Missing features", "missing": missing})
    vals = {f: flow.features[f] for f in RUNTIME.features}
    try:
        result = RUNTIME.predict(pd.DataFrame([vals], columns=RUNTIME.features), threshold=THRESHOLD)
    except Exception as e:
        raise HTTPException(422, f"Invalid feature payload: {e}") from e
    p = float(result["probability"][0]); label = "DDoS" if result["prediction"][0] else "BENIGN"
    factors = RUNTIME.top_factors(result["contrib"][0])
    review = bool(result["review_flag"][0]); ood_score = float(result["ood_score"][0])
    warnings = RUNTIME.range_warnings(vals)
    save([record(flow.source, label, p, factors, vals, review, ood_score)])
    return {"label": label, "prob_ddos": p, "confidence": float(result["confidence"][0]), "severity": severity(p),
            "threshold": THRESHOLD, "review_flag": review, "ood_score": None if not np.isfinite(ood_score) else ood_score,
            "factors": factors, "warnings": warnings, "model_version": MODEL_VERSION}

@app.post("/api/predict/batch", dependencies=[Depends(auth)])
async def predict_batch(file: UploadFile = File(...), source: str = Query("batch", max_length=64)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Upload a .csv file")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB")
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.columns = df.columns.str.strip()
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df)}); limit is {MAX_ROWS}")
    missing = [f for f in RUNTIME.features if f not in df.columns]
    if missing:
        raise HTTPException(422, {"message": "Missing columns", "missing": missing})
    X = df[RUNTIME.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = RUNTIME.predict(X, threshold=THRESHOLD)
    prob, pred = result["probability"], result["prediction"]
    flagged = np.where(pred == 1)[0]
    save([record(source, "DDoS" if pred[i] else "BENIGN", prob[i], RUNTIME.top_factors(result["contrib"][i], 5),
                X.iloc[i].where(X.iloc[i].notna(), None).to_dict(), result["review_flag"][i], result["ood_score"][i])
          for i in range(min(len(df), 5000))])
    out = {"rows": len(df), "ddos": int(pred.sum()), "benign": int(len(df) - pred.sum()),
           "high_severity": int((prob >= HIGH).sum()), "review_flagged": int(result["review_flag"].sum()),
           "ood_flagged": int(result["ood_flag"].sum()), "threshold": THRESHOLD,
           "invalid_cells": int(X.isna().sum().sum()),
           "model_version": MODEL_VERSION,
           "top_flagged": [{"row": int(i), "prob_ddos": float(prob[i])} for i in flagged[np.argsort(-prob[flagged])][:100]] if len(flagged) else []}
    label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if label_col:
        y = df[label_col].astype(str).str.strip().str.upper().eq("DDOS").to_numpy()
        pr = pred.astype(bool)
        tn, fp, fn, tp = map(int, confusion_matrix(y, pr, labels=[False, True]).ravel())
        out["vs_labels"] = {"accuracy": float((y == pr).mean()), "precision": tp/(tp+fp) if tp+fp else 0.0,
                             "recall": tp/(tp+fn) if tp+fn else 0.0, "f1": (2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "fpr": fp/(fp+tn) if fp+tn else 0.0, "fnr": fn/(fn+tp) if fn+tp else 0.0}
    return out

@app.get("/api/summary", dependencies=[Depends(auth)])
def summary(hours: int = Query(24, ge=1, le=720)):
    since = time.time() - hours * 3600
    bucket = 3600 if hours <= 48 else 86400
    with db() as c:
        tot = c.execute("SELECT label, severity, COUNT(*) n FROM predictions WHERE ts>=? GROUP BY label, severity", (since,)).fetchall()
        series = c.execute("SELECT CAST(ts/? AS INTEGER)*? t, label, COUNT(*) n FROM predictions WHERE ts>=? GROUP BY t, label ORDER BY t", (bucket, bucket, since)).fetchall()
        reviews = c.execute("SELECT COUNT(*) n FROM predictions WHERE ts>=? AND review_flag=1", (since,)).fetchone()[0]
    return {"hours": hours, "totals": [dict(r) for r in tot], "series": [dict(r) for r in series], "review_flagged": int(reviews), "model_version": MODEL_VERSION}

def _alert_query(limit, severity_f, label_f):
    q, a = "SELECT id,ts,source,label,prob,severity,model_version,review_flag,ood_score,factors FROM predictions WHERE 1=1", []
    if severity_f: q += " AND severity=?"; a.append(severity_f)
    if label_f: q += " AND label=?"; a.append(label_f)
    with db() as c:
        return c.execute(q + " ORDER BY id DESC LIMIT ?", (*a, limit)).fetchall()

@app.get("/api/alerts", dependencies=[Depends(auth)])
def alerts(limit: int = Query(100, ge=1, le=1000), severity_f: Optional[str] = Query(None, alias="severity", pattern="^(none|medium|high)$"), label: Optional[str] = Query(None, pattern="^(DDoS|BENIGN)$")):
    return [{**dict(r), "factors": json.loads(r["factors"] or "[]")} for r in _alert_query(limit, severity_f, label)]

@app.get("/api/alerts.csv", dependencies=[Depends(auth)])
def alerts_csv(limit: int = Query(1000, ge=1, le=10000)):
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["id","time_utc","source","label","prob_ddos","severity","model_version","review_flag","ood_score"])
    for r in _alert_query(limit, None, None):
        w.writerow([r["id"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["ts"])), r["source"], r["label"], round(r["prob"],6), r["severity"], r["model_version"], r["review_flag"], r["ood_score"]])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=alerts.csv"})

app.mount("/assets", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="assets")
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))
