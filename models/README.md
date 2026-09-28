# Model artifact policy

`model.joblib` is the **single canonical deployment artifact** used by both Streamlit and FastAPI.

`model_metadata.json` records the model version, schema version, SHA-256 checksum, threshold policy, and validation status.

`ood_isolation_forest_demo.joblib` is an **advisory OOD screening layer** trained on the bundled 6,000-row exploration sample. It is not a validated detector for new attack families.

When `python src/train.py` is run with the full CICIDS2017 DDoS CSV, it should overwrite `model.joblib`, `feature_schema.json`, and `model_metadata.json` together.
