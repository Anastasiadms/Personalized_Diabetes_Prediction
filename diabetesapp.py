import streamlit as st
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
import base64
from PIL import Image
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler_rf_diabetes.pkl')

# Helper function to compute BMI
def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

# Helper function to compute Total Risk Score
def calculate_risk_score(glucose, bmi, age, pregnancies):
    return round(glucose * 0.4 + bmi * 0.2 + age * 0.2 + pregnancies * 0.2, 2)

# PDF generator
def generate_pdf(data, prediction, bmi, risk_score):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="🩺 Diabetes Prediction Report", ln=1, align='C')
    pdf.ln(5)

    # Basic info
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Name: {data['Name']}", ln=1)
    pdf.cell(200, 10, txt=f"Age: {data['Age']}   Gender: {data['Gender']}", ln=1)
    pdf.cell(200, 10, txt=f"Pregnancies: {data['Pregnancies']}   DPF: {data['DPF']}", ln=1)

    # Health metrics
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="📋 Health Metrics", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Glucose: {data['Glucose']}     Insulin: {data['Insulin']}", ln=1)
    pdf.cell(200, 10, txt=f"Blood Pressure: {data['Blood Pressure']}     Skin Thickness: {data['Skin Thickness']}", ln=1)
    pdf.cell(200, 10, txt=f"Weight: {data['Weight (kg)']} kg     Height: {data['Height (cm)']} cm", ln=1)

    # Derived indicators
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="📊 Derived Indicators", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=1)
    pdf.cell(200, 10, txt=f"Total Risk Score: {risk_score}", ln=1)

    # Prediction result
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="🔍 Prediction Result", ln=1)
    pdf.set_font("Arial", '', 12)
    if prediction == 1:
        pdf.set_text_color(255, 0, 0)
        result_text = "Diabetic"
    else:
        pdf.set_text_color(0, 128, 0)
        result_text = "Non-Diabetic"
    pdf.cell(200, 10, txt=f"Prediction: {result_text}", ln=1)
    pdf.set_text_color(0, 0, 0)  # Reset

    # Symptoms
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="📝 Symptoms Checked", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=data['Symptoms Checked'])

    # Save file
    file_path = "diabetes_prediction_report.pdf"
    pdf.output(file_path)
    return file_path

# File download link
def download_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="diabetes_prediction_report.pdf">📥 Download Prediction Report</a>'
        return href

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Personalized Diabetes Prediction")

# User input
user_name = st.text_input("👤 Enter your name:", value="")
age = st.slider("Age", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.slider("Glucose", min_value=50, max_value=200, value=110)
skinthickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, step=0.1)
insulin = st.slider("Insulin", min_value=0.0, max_value=600.0, value=100.0)
dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, max_value=3.0, value=0.5)
bloodpressure = st.slider("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)

# Optional symptoms
with st.sidebar:
    st.markdown("### Optional Info")
    checklist = st.multiselect("Check if you have these symptoms:",
        ["Frequent urination", "Excessive thirst", "Fatigue", "Blurred vision"])

# Prediction trigger
if st.button("🔍 Predict"):
    bmi = calculate_bmi(weight, height)
    risk_score = calculate_risk_score(glucose, bmi, age, pregnancies)
    glucose_bmi = glucose * bmi
    insulin_log = np.log1p(insulin)
    dpf_log = np.log1p(dpf)
    bp_deviation = abs(bloodpressure - 80)

    # One-hot encodings for AgeGroup and BMICategory
    age_groups = {'AgeGroup_31-40': 0, 'AgeGroup_41-50': 0, 'AgeGroup_51-60': 0, 'AgeGroup_60+': 0}
    if 31 <= age <= 40:
        age_groups['AgeGroup_31-40'] = 1
    elif 41 <= age <= 50:
        age_groups['AgeGroup_41-50'] = 1
    elif 51 <= age <= 60:
        age_groups['AgeGroup_51-60'] = 1
    elif age > 60:
        age_groups['AgeGroup_60+'] = 1

    bmi_cats = {'BMICategory_Obese': 0, 'BMICategory_Overweight': 0, 'BMICategory_Underweight': 0}
    if bmi < 18.5:
        bmi_cats['BMICategory_Underweight'] = 1
    elif 25 <= bmi < 30:
        bmi_cats['BMICategory_Overweight'] = 1
    elif bmi >= 30:
        bmi_cats['BMICategory_Obese'] = 1

    input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bloodpressure,
        'SkinThickness': skinthickness,
        'BMI': bmi,
        'Age': age,
        'Insulin_log': insulin_log,
        'DPF_log': dpf_log,
        'BP_Deviation': bp_deviation,
        'Glucose_BMI': glucose_bmi,
        'Total_Risk_Score': risk_score,
        'is_obese': int(bmi >= 30),
        'is_high_glucose': int(glucose >= 140),
        'is_high_bp': int(bloodpressure >= 90),
        'is_high_insulin': int(insulin >= 25),
        'is_high_dpf': int(dpf_log >= np.percentile([np.log1p(dpf)], 80))
    }
    input_dict.update(age_groups)
    input_dict.update(bmi_cats)

    # Ensure correct order of columns
    ordered_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age',
                    'AgeGroup_31-40', 'AgeGroup_41-50', 'AgeGroup_51-60', 'AgeGroup_60+',
                    'BMICategory_Obese', 'BMICategory_Overweight', 'BMICategory_Underweight',
                    'Insulin_log', 'DPF_log', 'BP_Deviation', 'Glucose_BMI', 'Total_Risk_Score',
                    'is_obese', 'is_high_glucose', 'is_high_bp', 'is_high_insulin', 'is_high_dpf']

    input_df = pd.DataFrame([input_dict])[ordered_cols]
    data_scaled = scaler.transform(input_df)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][prediction]

    st.subheader("🧾 Prediction Result")
    st.write(f"**Patient Name:** {user_name if user_name else 'N/A'}")
    st.write("**Prediction:**", "🟥 Diabetic" if prediction == 1 else "🟩 Non-Diabetic")
    st.write("**BMI:**", bmi)
    st.write("**Total Risk Score:**", risk_score)
    st.write("**Confidence:**", f"{round(probability * 100, 2)}%")

    if prediction == 1:
        st.warning("⚠️ This result suggests a higher risk of diabetes. Please consult a healthcare provider for further testing.")
    else:
        st.success("✅ This result suggests a lower risk of diabetes.")


    user_info = {
        "Name": user_name,
        "Age": age,
        "Gender": gender,
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Skin Thickness": skinthickness,
        "Weight (kg)": weight,
        "Height (cm)": height,
        "Insulin": insulin,
        "Blood Pressure": bloodpressure,
        "DPF": dpf,
        "Symptoms Checked": ', '.join(checklist) if checklist else "None"
    }

    pdf_file = generate_pdf(user_info, prediction, bmi, risk_score)
    st.markdown(download_pdf(pdf_file), unsafe_allow_html=True)

