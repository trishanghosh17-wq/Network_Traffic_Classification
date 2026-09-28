# Model governance contract

Every deployed model must have:

- a unique model version;
- a SHA-256 artifact checksum;
- a feature-schema version;
- an explicit threshold;
- dataset/source documentation;
- internal and external evaluation results separated;
- a rollbackable artifact;
- a statement of known limitations.

`models/model.joblib` is the single canonical artifact used by Streamlit and FastAPI.

The current bundled artifact is an internal benchmark model. It must not be treated as production-approved until it has been tested on independent network/capture data.
