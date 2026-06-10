%%writefile README.md
# 🔮 Employee Attrition Predictor

A machine learning web app that predicts employee attrition risk using Logistic Regression.

## 📊 Dataset
- IBM HR Analytics Dataset (1,470 records, 35 features)
- 16% attrition rate (imbalanced — handled via class_weight='balanced')

## 🛠️ Tech Stack
- Python · Scikit-learn · Pandas · Streamlit
- Google Colab · GitHub

## 🤖 Model Performance
| Model | Accuracy | Attrition Detection |
|---|---|---|
| Logistic Regression (Balanced) | 71% | 62% ✅ |

## 🚀 How to Run
pip install -r requirements.txt
streamlit run app.py

## 💡 Key Findings
- Employees with longer tenure → lower attrition
- Low job satisfaction + no promotion → high attrition risk
- Overtime workers surprisingly stay longer