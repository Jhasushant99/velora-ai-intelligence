import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    drop_cols = ['id_student', 'final_result', 'date_unregistration']
    df_ml = df.drop(columns=drop_cols)

    cat_cols = [
        'code_module', 'code_presentation', 'gender', 'region',
        'highest_education', 'imd_band', 'age_band', 'disability'
    ]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    X = df_ml.drop(columns=['target_completion', 'is_high_risk'])
    y = df_ml['target_completion']

    return df, X, y, encoders


def train_models(X, y):
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    if HAS_XGB:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        xgb = GradientBoostingClassifier()

    xgb.fit(X_train, y_train)
    return rf, xgb


def train_dbscan(X):
    cluster_features = ['avg_score', 'total_clicks', 'assessments_attempted']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[cluster_features])

    dbscan = DBSCAN(eps=0.5, min_samples=20)
    dbscan.fit(X_scaled)

    return dbscan, scaler


if __name__ == "__main__":
    df, X, y, encoders = preprocess_data("data/features.csv")

    rf_model, xgb_model = train_models(X, y)
    dbscan_model, cluster_scaler = train_dbscan(X)

    FEATURES = X.columns.tolist()

    print("Saving models...")

    joblib.dump(rf_model, "models/random_forest_model.joblib")
    joblib.dump(xgb_model, "models/xgboost_model.joblib")
    joblib.dump(dbscan_model, "models/dbscan_model.joblib")
    joblib.dump(cluster_scaler, "models/cluster_scaler.joblib")
    joblib.dump(encoders, "models/label_encoders.joblib")
    joblib.dump(FEATURES, "models/features_list.joblib")

    print("✅ All models saved successfully")
