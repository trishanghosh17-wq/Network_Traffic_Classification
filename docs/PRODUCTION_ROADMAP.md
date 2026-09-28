# Production roadmap

## Phase 1 — Scientific reliability
1. Exact duplicate feature removal before splitting.
2. Feature selection inside CV.
3. Group-aware holdout.
4. Report PR-AUC, FPR, FNR and threshold curves.
5. Compare native importance, permutation importance and SHAP.
6. Validate on an independent capture/network.

## Phase 2 — Security capability
1. Add all available CICIDS2017 attack families.
2. Add unknown/OOD detection.
3. Measure per-class false negatives.
4. Add threshold/cost calibration.

## Phase 3 — Real-time product
Network telemetry → flow extraction → Kafka/MQTT → feature validation → model inference → risk score → SIEM/SOC alert → monitoring/drift → model registry/retraining.

## Phase 4 — Enterprise controls
- PostgreSQL instead of local SQLite for shared audit state
- authentication/authorization and secret management
- Dockerized services
- CI tests and model/data versioning
- latency/throughput monitoring
- rollbackable model releases
