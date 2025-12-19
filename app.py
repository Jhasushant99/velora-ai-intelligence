import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ---------------------------------
# 1. Designer UI Configuration
# ---------------------------------
st.set_page_config(page_title="Velora AI | Enhanced Intelligence", page_icon="💎", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #0F172A !important; padding-top: 2.5rem !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    div[data-testid="stMetric"] { background-color: #F8FAFC !important; border: 2px solid #CBD5E1 !important; border-radius: 12px !important; min-height: 140px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], h1, h2, h3, h4, p, label, .stMarkdown { color: #1E293B !important; font-weight: 700 !important; }
    .stButton>button { background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%); color: white !important; border-radius: 10px; height: 3.5em; font-weight: bold; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------
# 2. Optimized Model Loader
# ---------------------------------
@st.cache_resource
def load_velora_engine():
    try:
        path = "models/" if os.path.exists("models") else ""
        return {
            "rf": joblib.load(f"{path}random_forest_model.joblib"),
            "xgb": joblib.load(f"{path}xgboost_model.joblib"),
            "dbscan": joblib.load(f"{path}dbscan_model.joblib"),
            "scaler": joblib.load(f"{path}cluster_scaler.joblib"),
            "encoders": joblib.load(f"{path}label_encoders.joblib"),
            "features": joblib.load(f"{path}features_list.joblib")
        }
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        st.stop()

arts = load_velora_engine()

# ---------------------------------
# 3. Enhanced Analytics Pipeline
# ---------------------------------
def run_enhanced_pipeline(df):
    # Data Normalization
    for col in ["gender", "age_band", "highest_education", "disability"]:
        if col not in df.columns: df[col] = "Unknown"
    
    encoded_df = df.copy()
    for col, enc in arts['encoders'].items():
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].apply(lambda x: enc.transform([str(x)])[0] if str(x) in enc.classes_ else 0)
    
    final_X = encoded_df.reindex(columns=arts['features'], fill_value=0)
    
    # 1. Supervised Predictions (Completion/Dropout)
    rf_conf = arts['rf'].predict_proba(final_X)[:, 1] * 100
    xgb_conf = arts['xgb'].predict_proba(final_X)[:, 1] * 100
    df['rf_conf'], df['xgb_conf'] = rf_conf, xgb_conf
    
    engine_pref = rf_conf if model_choice == "Random Forest" else xgb_conf
    df['prediction'] = [1 if (score >= 70 or prob >= 40) else 0 for score, prob in zip(df['avg_score'], engine_pref)]
    
    # 2. Risk Flags (DBSCAN Behavioral Anomaly)
    scaled_data = arts['scaler'].transform(df[["avg_score", "total_clicks", "assessments_attempted"]])
    df['cluster'] = arts['dbscan'].fit_predict(scaled_data)
    
    def calculate_risk(row):
        if row['avg_score'] >= 75: return "🟢 Stable"
        if row['avg_score'] < 50 or row['cluster'] == -1: return "🚩 High Risk"
        return "🟡 Marginal"
    
    df['risk_status'] = df.apply(calculate_risk, axis=1)
    
    # 3. Chapter Difficulty & Effort
    df['difficulty'] = df.apply(lambda r: "🔴 High" if (r['total_clicks'] > 1100 and r['avg_score'] < 60) else "🟢 Standard", axis=1)
    df['effort'] = df.apply(lambda r: "⚡ Intensive" if r['total_clicks'] > 1200 else "✅ Balanced", axis=1)
    
    return df

# ---------------------------------
# 4. Interface (Sidebar & Tabs)
# ---------------------------------
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True) 
    st.image("https://cdn-icons-png.flaticon.com/512/3449/3449692.png", width=80)
    st.title("Velora AI")
    model_choice = st.radio("⚡ Prediction Engine", ["Random Forest", "XGBoost"])
    st.info("💡 **Logic Update:** Scores $\geq$ 70% automatically override AI dropout predictions for accuracy.")

st.title("💎 Velora Learning Intelligence Hub")
tab1, tab2, tab3 = st.tabs(["🎯 Assessment", "📂 Batch Analysis", "📘 Guide"])

with tab1:
    st.subheader("🧑‍🎓 Individual Student Deep-Dive")
    c1, c2, c3 = st.columns(3)
    with c1: sid = st.text_input("Student ID", "STU-882")
    with c1: score = st.slider("Avg Score %", 0, 100, 85)
    with c2: clicks = st.number_input("Engagement (Clicks)", 0, 5000, 900)
    with c2: attempts = st.number_input("Assessment Attempts", 1, 15, 8)
    with c3: chapter = st.number_input("Chapter Order", 1, 30, 5)
    with c3: dis = st.selectbox("Disability Status", ["No", "Yes"])

    if st.button("🚀 Execute Pipeline"):
        raw = pd.DataFrame([{"id_student": sid, "avg_score": score, "total_clicks": clicks, "assessments_attempted": attempts, "chapter_order": chapter, "gender": "Male", "age_band": "35-55", "highest_education": "High School", "disability": dis}])
        res = run_enhanced_pipeline(raw)
        
        # 4. Summary Insights
        st.subheader("📊 Algorithmic Validation Table")
        disp_df = res[['id_student', 'prediction', 'rf_conf', 'xgb_conf', 'risk_status', 'difficulty']]
        disp_df['prediction'] = disp_df['prediction'].map({1: "✅ PASS", 0: "❌ RISK"})
        st.table(disp_df.style.format({"rf_conf": "{:.1f}%", "xgb_conf": "{:.1f}%"}))

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", disp_df['prediction'].iloc[0])
        m2.metric("Risk Level", res['risk_status'].iloc[0])
        m3.metric("Effort Level", res['effort'].iloc[0])
        m4.metric("Difficulty", res['difficulty'].iloc[0])

with tab2:
    st.subheader("📂 Bulk Intelligence Mode")
    up_csv = st.file_uploader("Upload CSV Data", type="csv")
    if up_csv:
        bulk_raw = pd.read_csv(up_csv)
        if st.button("🔍 Run Bulk Analysis"):
            processed = run_enhanced_pipeline(bulk_raw)
            processed['prediction'] = processed['prediction'].map({1: "✅ PASS", 0: "❌ RISK"})
            cols = ['id_student', 'prediction', 'rf_conf', 'xgb_conf', 'risk_status', 'effort', 'difficulty']
            st.dataframe(processed[cols].style.format({"rf_conf": "{:.1f}%", "xgb_conf": "{:.1f}%"}), use_container_width=True)

with tab3:
    st.subheader("📘 Intelligence Architecture & Definitions")
    
    st.markdown("""
    - **Predictions**: Completion vs. Dropout based on a hybrid logic of AI confidence and academic thresholds.
    - **Risk Flags**: Multi-level status (**Stable**, **Marginal**, **High Risk**) triggered by failing grades or behavioral outliers.
    - **Chapter Difficulty**: Flags frustration points where students exhibit high clicks but maintain low scores.
    - **Effort Detection**: Categorizes students as 'Intensive' or 'Balanced' to prevent potential burnout.
    """)

st.divider()
st.caption("Developed by Sushant Shekhar | Velora AI 2025")