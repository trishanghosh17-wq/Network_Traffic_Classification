# AI-Based DDoS Detection Using Network Flow Data

A company-oriented machine-learning prototype for **BENIGN vs DDoS** network-flow detection, with model governance, exact TreeSHAP explanations, advisory OOD screening, threshold analysis, external-validation tooling, FastAPI serving, Streamlit UI, drift checks and CI.

## Current scope

The bundled model is **binary: BENIGN vs DDoS**. It is **not** a general multi-attack IDS. A genuine multi-attack model requires adding the other CICIDS2017 attack-family captures under `data/multiclass/` and running `python src/train_multiclass.py`.

## What is improved

- One canonical model artifact: `models/model.joblib`.
- One feature contract: `models/feature_schema.json`.
- Model checksum/version metadata: `models/model_metadata.json`.
- Leakage-aware binary training with exact duplicate-feature removal and in-pipeline mutual-information selection.
- PR-AUC, FPR, FNR and threshold analysis.
- Native XGBoost, permutation and SHAP global importance views.
- Exact TreeSHAP contributions for single-flow explanations.
- Advisory OOD screening for atypical flows.
- External labeled-data evaluation workflow.
- Model-only latency/throughput benchmark.
- FastAPI and Streamlit use the same canonical artifact.
- API health endpoint exposes model version and checksum prefix.
- API audit records include model version, review/OOD flags and factors.
- GitHub Actions CI for syntax and tests.

## Current benchmark

The bundled internal holdout benchmark is:

- Accuracy: **99.9955%**
- ROC-AUC: **99.9980%**
- PR-AUC: **99.9985%**
- Precision: **100.0000%**
- Recall: **99.9922%**
- F1: **99.9961%**
- FPR: **0.000000%**
- FNR: **0.007812%**

These are **internal benchmark results**, not a production-accuracy claim. The bundled source does not provide trustworthy timestamp/session IDs for a true temporal split, and no independent network/capture test is bundled.

## Why two features dominate native importance

The model uses many flow features, but native XGBoost split gain is highly concentrated in two packet/segment features. Several CICFlowMeter variables are redundant or mathematically equivalent in the bundled sample. The repository therefore reports native importance alongside permutation importance, SHAP global importance and exploratory feature ablation.

## Reproducibility

The full raw dataset is not committed. Place the original DDoS CICIDS2017 CSV at:

`data/DDos.pcap_ISCX.csv`

Then run:

`python src/train.py`

The training script writes the canonical model and its schema/metadata together.

## External validation

Use the Streamlit **External Test** tab or:

`python src/evaluate_external.py path/to/external.csv`

The external dataset must contain the same feature definitions plus `Label` with BENIGN/DDOS for the current binary model.

## Multi-attack training

Put the complete labeled attack-family files in:

`data/multiclass/`

Then run:

`python src/train_multiclass.py`

The script creates `models/multiclass_model.joblib` and `results/multiclass_metrics.json`. This is the genuine multiclass path; the current Streamlit app deliberately does not pretend that the binary model already handles those classes.

## Run Streamlit

`streamlit run app/streamlit_app.py`

Streamlit Cloud main file:

`app/streamlit_app.py`

## Run API

`DDOS_API_KEY=change-me uvicorn backend.main:app --port 8000`

## Engineering notes

`docs/ARCHITECTURE.md` — production path

`docs/MODEL_GOVERNANCE.md` — artifact/version contract

`docs/EXTERNAL_VALIDATION.md` — independent-test protocol

`docs/CANDIDATE_REVIEW.md` — company/interview review criteria
