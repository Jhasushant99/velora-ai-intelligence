##🎯 Project Overview
Velora is an AI-driven educational tool designed to analyze student profiles and predict academic risk levels. By leveraging machine learning, it identifies students who may need additional support based on their engagement metrics, demographics, and assessment performance.
##🚀 Key Features
A.Student Profile Analysis: Processes data including engagement clicks, average scores, and demographics.
B.Dual-Model Prediction: Utilizes both Random Forest and XGBoost for high-confidence risk assessment.
C.Risk Status Categorization: Real-time flagging of "High Risk" students.
D.Material Difficulty Insights: Analyzes if the current material is standard or too difficult for the learner.
##📊 Algorithms Used
1. Random Forest (RF)
Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees. It is highly robust against overfitting and handles categorical data effectively.

2. XGBoost (Extreme Gradient Boosting)
XGBoost is a powerful gradient-boosting framework. It is used in Velora for its speed and performance in predicting student churn or failure based on complex engagement patterns.
🛠️ Technical Stack
#Language: Python

#Libraries: Scikit-Learn, XGBoost, Pandas, NumPy

#Frontend/UI: (e.g., Streamlit or Flask - update as per your project)

#AI Models: Random Forest Classifier, XGBoost Classifier
Comparative Intelligence TableThe hub provides a detailed breakdown of model confidence levels:Student IDPredictionRF ConfidenceXGB ConfidenceRisk StatusDifficulty as follows-> STU-882,❌ RISK,55.0%,79.2%,🚩 High Risk,🟢 Standard
⚙️ Installation & Usage
#Clone the repository:
git clone https://github.com/your-username/velora-student-intelligence.git

#Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py


🧑‍🎓 Author
Developed by Sushant Shekhar Velora AI 2025
