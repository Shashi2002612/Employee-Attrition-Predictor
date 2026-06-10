import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Column names manually
FEATURES = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
            'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
            'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
            'OverTime', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

st.title("🔮 Employee Attrition Predictor")
st.write("Enter employee details to predict attrition risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

with col2:
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_manager = st.slider("Years With Current Manager", 0, 17, 3)
    num_companies = st.slider("Num Companies Worked", 0, 9, 2)
    total_working_years = st.slider("Total Working Years", 0, 40, 8)

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[
        age, 2, 500, 1, 3,
        2, 1, environment_satisfaction, 0,
        50, 2, 2, 2, job_satisfaction,
        2, monthly_income, 5000, num_companies,
        1 if overtime == "Yes" else 0,
        15, 3, 3, 0,
        total_working_years, 3, 3, years_at_company,
        3, years_since_promotion, years_with_manager
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk! ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Attrition Risk ({probability*100:.1f}% probability)")

    st.progress(float(probability))
