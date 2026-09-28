"""DDoS Traffic Intelligence — company-oriented Streamlit application.

Run: streamlit run app/streamlit_app.py
Canonical model: models/model.joblib
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.runtime import ModelRuntime

st.set_page_config(page_title="DDoS Traffic Intelligence", page_icon="🛡️", layout="wide")

@st.cache_resource(show_spinner=False)
def load_runtime():
    return ModelRuntime()

@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = ROOT / path
    if p.exists():
        with p.open() as f: return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_csv(path: str):
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

try:
    runtime = load_runtime()
except Exception as exc:
    st.error(f"Model runtime could not start: {exc}")
    st.stop()

features = runtime.features
metrics = load_json("results/final_metrics.json")
meta = runtime.metadata
feat_imp = load_csv("results/final_feature_importance.csv").rename(columns={"Unnamed: 0":"Feature"})
mi_train = load_csv("results/feature_selection_mutual_information.csv").rename(columns={"Unnamed: 0":"Feature"})
perm_imp = load_csv("results/permutation_importance_sample.csv")
shap_imp = load_csv("results/shap_importance_sample.csv")
class_dist = load_csv("results/class_distribution.csv")
dataset_audit = load_csv("results/dataset_audit.csv")
misclassified = load_csv("results/misclassified_flows.csv")
thresh_df = load_csv("results/threshold_analysis_holdout.csv")
ablation_df = load_csv("results/feature_ablation_exploratory_sample.csv")
bench = load_json("results/inference_benchmark.json")
sample_path = ROOT / "data" / "eda_sample.csv"
sample_df = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()

# ---------------- helpers ----------------
def severity(p: float, threshold: float = 0.5, high: float = 0.9) -> str:
    return "HIGH" if p >= high else "MEDIUM" if p >= threshold else "NONE"

def exact_factors(frame: pd.DataFrame, n=10):
    r = runtime.predict(frame, threshold=0.5)
    row = r["contrib"][0]
    idx = np.argsort(-np.abs(row))[:n]
    return pd.DataFrame({"Feature":[features[i] for i in idx], "SHAP contribution (log-odds)":[row[i] for i in idx]})

def evaluate_external(df: pd.DataFrame, threshold: float):
    label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if not label_col:
        return None, "CSV must contain a Label column for external evaluation."
    result = runtime.predict(df, threshold=threshold)
    y = df[label_col].astype(str).str.strip().str.upper().eq("DDOS").astype(int).to_numpy()
    p = result["probability"]
    pred = result["prediction"]
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out = {
        "Rows": len(df), "Accuracy": accuracy_score(y,pred), "Precision": precision_score(y,pred,zero_division=0),
        "Recall": recall_score(y,pred,zero_division=0), "F1": f1_score(y,pred,zero_division=0),
        "ROC-AUC": roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        "PR-AUC": average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,
        "FPR": fp/(fp+tn) if fp+tn else 0.0, "FNR": fn/(fn+tp) if fn+tp else 0.0,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    return out, None

# ---------------- header ----------------
st.title("🛡️ AI-Based DDoS Detection Using Network Flow Data")
st.caption(f"Binary production prototype: BENIGN vs DDoS • Model {meta.get('model_version','—')} • Schema {meta.get('feature_schema_version','—')}")
st.info("**Important:** This bundled model is a DDoS-specific binary classifier. It is not yet a validated multi-attack IDS. The project now includes an engineering-ready multiclass training path, external-validation workflow, exact TreeSHAP explanations, threshold analysis, OOD screening and model-version controls.")

with st.sidebar:
    st.header("Deployment status")
    c1, c2 = st.columns(2)
    c1.metric("Features", len(features))
    c2.metric("CPU cores", os.cpu_count() or 1)
    st.caption(f"Artifact SHA-256: `{runtime.model_sha256[:16]}…`")
    st.caption(f"Default threshold: **{meta.get('threshold_default',0.5):.2f}**")
    candidate = meta.get('threshold_candidate_from_internal_holdout')
    if candidate is not None:
        st.caption(f"Internal-holdout candidate threshold: **{candidate:.2f}**")
    st.divider()
    st.write("**Validation status**")
    st.warning("Internal benchmark only. External network/capture validation is still required.")
    st.write("**OOD status**")
    st.info("Advisory screening only; not a validated unseen-attack detector.")

# ---------------- tabs ----------------
tab_overview, tab_explore, tab_single, tab_batch, tab_insights, tab_validation, tab_external, tab_quality = st.tabs([
    "📊 Overview","🔍 Explore","🎯 Single Flow","📁 Batch","🧠 Explainability","📈 Validation","🌐 External Test","🧪 Quality & Drift"
])

# Overview
with tab_overview:
    st.subheader("Current model performance")
    if metrics:
        cols=st.columns(6)
        for c, label, key in zip(cols,["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"],["Accuracy","Precision","Recall","F1","ROC_AUC","PR_AUC"]):
            c.metric(label,f"{metrics.get(key,0):.4%}")
        st.caption("These numbers are an internal group-aware holdout benchmark, not independent production performance.")
    left,right=st.columns(2)
    with left:
        st.subheader("Holdout confusion matrix")
        if metrics:
            cm=np.array([[metrics['TN'],metrics['FP']],[metrics['FN'],metrics['TP']]])
            fig=px.imshow(cm,text_auto=True,x=["Pred BENIGN","Pred DDoS"],y=["Actual BENIGN","Actual DDoS"],color_continuous_scale="Blues")
            fig.update_layout(height=350,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Class distribution")
        if not class_dist.empty:
            fig=px.pie(class_dist,names="Label",values="Count",hole=.55)
            fig.update_layout(height=350)
            st.plotly_chart(fig,use_container_width=True)
    st.subheader("Operational interpretation")
    k1,k2,k3,k4=st.columns(4)
    k1.metric("False positives", int(metrics.get('FP',0)))
    k2.metric("False negatives", int(metrics.get('FN',0)))
    k3.metric("FPR", f"{metrics.get('False_Positive_Rate',0):.4%}")
    k4.metric("FNR", f"{metrics.get('False_Negative_Rate',0):.4%}")
    st.markdown("**Company question:** how does this behave on a completely different capture? Use the **External Test** tab to evaluate a labeled CSV from another environment.")

# Explore
with tab_explore:
    st.caption("Bundled 6,000-flow sample is for exploration only. It is not an external validation set.")
    if sample_df.empty:
        st.warning("data/eda_sample.csv is missing.")
    else:
        f=st.selectbox("Feature",features)
        log_scale=st.checkbox("Log1p transform",True)
        p=sample_df[[f,"Label"]].copy()
        if log_scale: p[f]=np.log1p(p[f].clip(lower=0))
        fig=px.histogram(p,x=f,color="Label",barmode="overlay",nbins=60,opacity=.65)
        fig.update_layout(height=420)
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Redundancy awareness")
        dup=load_csv("results/exact_duplicate_feature_pairs_in_sample.csv")
        if not dup.empty:
            st.dataframe(dup,use_container_width=True)

# Single flow
with tab_single:
    st.subheader("Single-flow prediction")
    st.caption("The explanation below uses the model's actual XGBoost TreeSHAP log-odds contributions, not a hand-built feature-distance heuristic.")
    if "flow_vals" not in st.session_state:
        st.session_state.flow_vals={f:float(runtime.stats.get(f,{}).get('median_overall',0.0)) for f in features}
    b1,b2,b3=st.columns([1,1,2])
    if b1.button("Load typical BENIGN"):
        st.session_state.flow_vals={f:float(runtime.stats.get(f,{}).get('median_benign',0.0)) for f in features}
    if b2.button("Load typical DDoS"):
        st.session_state.flow_vals={f:float(runtime.stats.get(f,{}).get('median_ddos',0.0)) for f in features}
    threshold=st.slider("Decision threshold",0.05,0.95,float(meta.get('threshold_default',0.5)),0.01)
    vals={}
    cols=st.columns(3)
    for i,f in enumerate(features):
        s=runtime.stats.get(f,{})
        lo=float(s.get('min',0)); hi=float(s.get('max',lo+1));
        if hi<=lo: hi=lo+1
        default=float(st.session_state.flow_vals.get(f,s.get('median_overall',lo)))
        vals[f]=cols[i%3].number_input(f,min_value=lo,max_value=hi,value=float(np.clip(default,lo,hi)),format="%.4f",key=f"sf_{f}")
        st.session_state.flow_vals[f]=vals[f]
    if st.button("Analyze flow",type="primary"):
        frame=pd.DataFrame([vals])
        res=runtime.predict(frame,threshold=threshold)
        p=float(res['probability'][0]); label="DDoS" if res['prediction'][0] else "BENIGN"
        review=bool(res['review_flag'][0]); ood=float(res['ood_score'][0])
        a,b,c=st.columns(3)
        a.metric("Prediction",label)
        b.metric("DDoS probability",f"{p:.2%}")
        c.metric("Confidence",f"{res['confidence'][0]:.2%}")
        if review: st.warning("⚠️ Review recommended: the flow is low-confidence and/or atypical relative to the bundled reference distribution. This does not prove a new attack.")
        if np.isfinite(ood): st.caption(f"Advisory OOD decision score: {ood:.4f}")
        warnings=runtime.range_warnings(vals)
        if warnings: st.warning("\n".join(["**Range warnings:**"]+warnings[:8]))
        st.subheader("Exact TreeSHAP factors")
        factors=exact_factors(frame,10)
        fig=px.bar(factors.sort_values('SHAP contribution (log-odds)'),x='SHAP contribution (log-odds)',y='Feature',orientation='h')
        fig.update_layout(height=380)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(factors,use_container_width=True)

# Batch
with tab_batch:
    st.subheader("Batch prediction + operational alert volume")
    file=st.file_uploader("Upload a CSV containing the 30 feature columns (Label optional)",type=['csv'],key='batch')
    if file:
        df=pd.read_csv(file); df.columns=df.columns.str.strip()
        missing=[f for f in features if f not in df.columns]
        if missing: st.error(f"Missing required columns: {missing[:10]}{' …' if len(missing)>10 else ''}")
        else:
            threshold=st.slider("Batch threshold",0.05,0.95,float(meta.get('threshold_default',0.5)),0.01,key='batch_thresh')
            res=runtime.predict(df,threshold=threshold); p=res['probability']; pred=res['prediction']
            out=df.copy(); out['Prediction']=np.where(pred==1,'DDoS','BENIGN'); out['DDoS_Probability']=p; out['Review_Flag']=res['review_flag']; out['OOD_Flag']=res['ood_flag']
            a,b,c,d,e=st.columns(5)
            a.metric("Flows",f"{len(out):,}"); b.metric("DDoS",f"{int(pred.sum()):,}"); c.metric("Alerts / 10k",f"{int(pred.sum()/max(len(out),1)*10000):,}"); d.metric("Review flags",f"{int(res['review_flag'].sum()):,}"); e.metric("OOD flags",f"{int(res['ood_flag'].sum()):,}")
            if 'Label' in df.columns:
                scored,err=evaluate_external(df,threshold)
                if scored:
                    st.write("**Uploaded-label evaluation**")
                    met=pd.DataFrame([scored])
                    st.dataframe(met,use_container_width=True)
            fig=px.histogram(out,x='DDoS_Probability',color='Prediction',nbins=40,barmode='overlay')
            fig.add_vline(x=threshold,line_dash='dash')
            fig.update_layout(height=330)
            st.plotly_chart(fig,use_container_width=True)
            st.download_button("Download scored CSV",out.to_csv(index=False),"ddos_scored_flows.csv","text/csv")

# Explainability
with tab_insights:
    st.subheader("Feature importance: three complementary views")
    if not feat_imp.empty:
        top2=feat_imp.sort_values('importance',ascending=False).head(2)
        share=top2['importance'].sum()
        st.info(f"The current model uses {len(features)} features, but native XGBoost split gain assigns {share:.2%} of importance to the top two. Do not interpret this as proof that only two variables matter. The sample contains exact redundant CICFlowMeter features, and permutation/TreeSHAP tell a broader story.")
        top=feat_imp.sort_values('importance').tail(15)
        st.plotly_chart(px.bar(top,x='importance',y='Feature',orientation='h'),use_container_width=True)
    if not perm_imp.empty:
        st.markdown("### Permutation importance")
        st.dataframe(perm_imp.sort_values('permutation_importance_mean',ascending=False).head(15),use_container_width=True)
    if not shap_imp.empty:
        st.markdown("### SHAP global importance")
        top_s=shap_imp.sort_values('mean_abs_shap',ascending=False).head(15).sort_values('mean_abs_shap')
        st.plotly_chart(px.bar(top_s,x='mean_abs_shap',y='Feature',orientation='h'),use_container_width=True)
    st.markdown("### Exploratory feature ablation")
    if not ablation_df.empty:
        st.warning("These ablation numbers use the bundled 6,000-flow exploration sample and are not a replacement for independent validation.")
        st.dataframe(ablation_df.style.format({c:'{:.4%}' for c in ['Accuracy','Precision','Recall','F1','ROC_AUC','PR_AUC','FPR','FNR']}),use_container_width=True)
    if not misclassified.empty:
        with st.expander(f"Misclassified internal holdout flows ({len(misclassified)})"):
            st.dataframe(misclassified,use_container_width=True)

# Validation
with tab_validation:
    st.subheader("Validation, thresholds and production questions")
    if metrics:
        k=st.columns(5)
        k[0].metric('Accuracy',f"{metrics['Accuracy']:.4%}"); k[1].metric('PR-AUC',f"{metrics.get('PR_AUC',0):.4%}"); k[2].metric('FPR',f"{metrics.get('False_Positive_Rate',0):.4%}"); k[3].metric('FNR',f"{metrics.get('False_Negative_Rate',0):.4%}"); k[4].metric('Holdout flows',f"{int(metrics['TN']+metrics['FP']+metrics['FN']+metrics['TP']):,}")
    if not thresh_df.empty:
        st.markdown("### Threshold trade-off")
        st.dataframe(thresh_df.style.format({c:'{:.4%}' for c in ['Accuracy','Precision','Recall','F1','FPR','FNR']}),use_container_width=True,height=320)
        fig=go.Figure()
        for col in ['Precision','Recall','FPR','FNR']:
            fig.add_trace(go.Scatter(x=thresh_df['Threshold'],y=thresh_df[col],mode='lines+markers',name=col))
        fig.update_layout(height=380,xaxis_title='Threshold',yaxis_title='Rate')
        st.plotly_chart(fig,use_container_width=True)
        st.info(f"Candidate threshold from the current internal holdout: **{meta.get('threshold_candidate_from_internal_holdout','—')}**. Treat this as an analysis result, not a production approval.")
    st.markdown("### What is still missing for a security deployment?")
    st.write("Independent capture validation → true temporal/session split → external dataset test → shadow mode on live telemetry → analyst feedback → retraining/version approval.")

# External test
with tab_external:
    st.subheader("🌐 Independent labeled-data evaluation")
    st.caption("Use this tab with a labeled CSV from a different capture, network, or dataset. The CSV must contain the same 30 feature columns and a Label column with BENIGN/DDOS for the current binary model.")
    ext=st.file_uploader("Upload independent labeled CSV",type=['csv'],key='external')
    if ext:
        try:
            df=pd.read_csv(ext); df.columns=df.columns.str.strip()
            missing=[f for f in features if f not in df.columns]
            if missing: st.error(f"Missing required features: {missing[:10]}")
            else:
                threshold=st.slider("External evaluation threshold",0.05,0.95,float(meta.get('threshold_default',0.5)),0.01,key='ext_thresh')
                scored,err=evaluate_external(df,threshold)
                if err: st.error(err)
                else:
                    st.success("External evaluation completed on the uploaded file. This result should be reported separately from the internal benchmark.")
                    m=pd.DataFrame([scored]); st.dataframe(m,use_container_width=True)
                    cm=np.array([[scored['TN'],scored['FP']],[scored['FN'],scored['TP']]])
                    fig=px.imshow(cm,text_auto=True,x=['Pred BENIGN','Pred DDoS'],y=['Actual BENIGN','Actual DDoS'],color_continuous_scale='Blues'); fig.update_layout(height=360,coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)
        except Exception as exc:
            st.error(f"Could not evaluate the file: {exc}")

# Quality & drift
with tab_quality:
    st.subheader("🧪 Data Quality, OOD screening and Drift")
    if sample_df.empty:
        st.warning("Bundled sample unavailable.")
    else:
        numeric=sample_df[features].select_dtypes(include=np.number).columns.tolist()
        q=pd.DataFrame({'Feature':numeric,'Missing %':[100*sample_df[f].isna().mean() for f in numeric], 'Unique':[sample_df[f].nunique(dropna=True) for f in numeric]})
        st.dataframe(q.sort_values('Missing %',ascending=False).style.format({'Missing %':'{:.3f}%'}),use_container_width=True,height=300)
        mid=len(sample_df)//2; ref=sample_df.iloc[:mid]; cur=sample_df.iloc[mid:]
        def psi(a,b,bins=10):
            a=pd.to_numeric(a,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy(); b=pd.to_numeric(b,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            if len(a)<2 or len(b)<2 or np.min(a)==np.max(a): return np.nan
            qs=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
            if len(qs)<3:return np.nan
            edges=np.r_[[-np.inf],qs[1:-1],[np.inf]]; pa=np.histogram(a,edges)[0].astype(float); pb=np.histogram(b,edges)[0].astype(float); pa/=max(pa.sum(),1); pb/=max(pb.sum(),1); pa=np.clip(pa,1e-6,None); pb=np.clip(pb,1e-6,None); return float(np.sum((pb-pa)*np.log(pb/pa)))
        drift=pd.DataFrame({'Feature':numeric,'PSI':[psi(ref[f],cur[f]) for f in numeric]}).dropna().sort_values('PSI',ascending=False)
        st.markdown("### PSI drift (sample-half vs sample-half)")
        st.dataframe(drift,use_container_width=True)
        st.caption("This is a monitoring demonstration, not evidence of production drift. A production system should compare live windows with a versioned training/reference distribution.")
        if bench:
            st.markdown("### Local inference benchmark")
            a,b,c=st.columns(3); a.metric('Flows benchmarked',f"{bench.get('rows',0):,}"); b.metric('Median throughput',f"{bench.get('median_throughput_flows_per_sec',0):,.0f} flows/s"); c.metric('Median batch time',f"{bench.get('median_total_seconds',0):.3f} s")
            st.caption(bench.get('note',''))
        st.markdown("### Runtime contract")
        st.code(json.dumps({'model_version':meta.get('model_version'),'schema_version':meta.get('feature_schema_version'),'sha256':runtime.model_sha256,'features':len(features),'classes':meta.get('classes')},indent=2),language='json')
