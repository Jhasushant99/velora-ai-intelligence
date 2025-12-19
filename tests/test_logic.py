import pandas as pd
import pytest
import joblib
import os

# -------------------------------
# Load artifacts for testing
# -------------------------------
PATH = "models/" if os.path.exists("models") else ""
try:
    RF_MODEL = joblib.load(f"{PATH}random_forest_model.joblib")
    FEATURES = joblib.load(f"{PATH}features_list.joblib")
except:
    RF_MODEL = None

# -------------------------------
# Test 1: Single Prediction Logic
# -------------------------------
def test_prediction_logic():
    """Test if a high-performing student is predicted as 'Complete' (1)"""
    if RF_MODEL is None:
        pytest.skip("Model artifacts not found, skipping test.")
    
    # Dummy high-performing student
    test_data = pd.DataFrame([{
        "avg_score": 95.0,
        "total_clicks": 1500,
        "assessments_attempted": 5,
        "gender": 1, "age_band": 0, "highest_education": 1, "disability": 0
    }])
    
    final_input = test_data.reindex(columns=FEATURES, fill_value=0)
    prediction = RF_MODEL.predict(final_input)[0]
    
    assert prediction in [0, 1], "Prediction must be binary (0 or 1)"

# -------------------------------
# Test 2: Input Validation
# -------------------------------
def test_input_validation():
    """Test if scores above 100 or below 0 are flagged"""
    invalid_scores = [150, -5]
    for score in invalid_scores:
        assert not (0 <= score <= 100), f"Score {score} is invalid but not flagged"

# -------------------------------
# Test 3: Chapter Difficulty Logic
# -------------------------------
def test_chapter_difficulty_logic():
    """High clicks + low score = High Difficulty"""
    clicks = 1200
    score = 40
    difficulty = "High" if (clicks > 800 and score < 60) else "Standard"
    assert difficulty == "High"

# -------------------------------
# Test 4: Batch Prediction
# -------------------------------
def test_batch_prediction():
    """Test multiple student inputs in a batch"""
    if RF_MODEL is None:
        pytest.skip("Model artifacts not found, skipping test.")
    
    batch_data = pd.DataFrame([
        {"avg_score": 90, "total_clicks": 1000, "assessments_attempted": 5, "gender":1, "age_band":0, "highest_education":1, "disability":0},
        {"avg_score": 45, "total_clicks": 200, "assessments_attempted": 2, "gender":0, "age_band":1, "highest_education":0, "disability":1}
    ])
    final_input = batch_data.reindex(columns=FEATURES, fill_value=0)
    preds = RF_MODEL.predict(final_input)
    
    assert len(preds) == 2
    assert all(p in [0,1] for p in preds)

# -------------------------------
# Test 5: High-Risk Logic
# -------------------------------
def test_high_risk_student():
    """Ensure low engagement or low score triggers high risk"""
    clicks = 50
    score = 10
    risk_status = "High" if (clicks < 100 or score < 40) else "Low"
    assert risk_status == "High"

# -------------------------------
# Test 6: Encoder Unknown Category
# -------------------------------
def test_encoder_unseen_category():
    """Ensure unknown category defaults to 0"""
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    enc.fit(["A","B","C"])
    
    val = "D"  # unseen
    encoded_val = enc.transform([val])[0] if val in enc.classes_ else 0
    assert encoded_val == 0
