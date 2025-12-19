import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
import xgboost as xgb
import joblib
import os

# ---------------------------
# 1. Dummy dataset
# ---------------------------
X = pd.DataFrame({
    "avg_score": np.random.randint(0, 100, 100),
    "total_clicks": np.random.randint(0, 5000, 100),
    "assessments_attempted": np.random.randint(1, 15, 100),
    "gender": np.random.choice(["Male", "Female"], 100),
    "age_band": np.random.choice(["18-25", "26-35", "36-45", "46-55"], 100),
    "highest_education": np.random.choice(["High School", "Bachelor", "Master"], 100),
    "disability": np.random.choice(["Yes", "No"], 100)
})
y = np.random.randint(0, 2, 100)

# ---------------------------
# 2. Encode categorical features
# ---------------------------
encoders = {}
for col in ["gender", "age_band", "highest_education", "disability"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ---------------------------
# 3. Train Random Forest
# ---------------------------
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)

# ---------------------------
# 4. Train XGBoost
# ---------------------------
xgb_model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X, y)

# ---------------------------
# 5. DBSCAN clustering
# ---------------------------
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X[["avg_score", "total_clicks", "assessments_attempted"]])
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(scaled_X)

# ---------------------------
# 6. Save all models and encoders
# ---------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/random_forest_model.joblib")
joblib.dump(xgb_model, "models/xgboost_model.joblib")
joblib.dump(dbscan, "models/dbscan_model.joblib")
joblib.dump(scaler, "models/cluster_scaler.joblib")
joblib.dump(encoders, "models/label_encoders.joblib")
joblib.dump(list(X.columns), "models/features_list.joblib")

print("✅ Dummy models created successfully in 'models/' folder")
