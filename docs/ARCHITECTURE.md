# Company-oriented architecture

## Current working path

Network-flow CSV or manually supplied flow features
→ schema/range validation
→ canonical `models/model.joblib`
→ DDoS probability
→ exact TreeSHAP factors
→ advisory OOD screen
→ severity / review flag
→ Streamlit or FastAPI output
→ auditable prediction record

## Production target

Network sensor / flow extractor
→ Kafka or MQTT
→ feature validation
→ model gateway
→ known-attack classifier
→ unknown/OOD detector
→ risk scoring
→ SIEM/SOC
→ feedback + monitoring
→ approved model retraining

The current repository implements the model gateway, explainability, advisory OOD screen, audit logging, threshold analysis and monitoring demonstrations. It does not claim that the live network-ingestion or SIEM components are already deployed.
