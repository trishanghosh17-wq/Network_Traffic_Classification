# Streamlit Cloud deployment

1. Push the repository contents (not the ZIP itself) to GitHub.
2. On Streamlit Community Cloud, choose the repository and `main` branch.
3. Set the main file to `app/streamlit_app.py`.
4. The app uses the canonical `models/model.joblib`; no local FastAPI service is required for the Streamlit dashboard.
5. After the first successful deployment, copy the generated `https://<app-name>.streamlit.app` URL into your README/PPT.

The current app is a binary DDoS detector. Do not describe the deployed app as a multi-attack IDS until `src/train_multiclass.py` has been run with the full attack-family data and the multiclass model has been independently evaluated.
