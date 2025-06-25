import streamlit as st
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64

# Load model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler_rf_diabetes.pkl')

# Helper functions
def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

def calculate_risk_score(glucose, bmi, age, pregnancies):
    return round(glucose * 0.4 + bmi * 0.2 + age * 0.2 + pregnancies * 0.2, 2)

def generate_pdf(data, prediction, bmi, risk_score):
    pdf = FPDF()
    pdf.add_page()
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M')

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated on: {report_date}", ln=1, align='R')
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Name: {data['Name']}", ln=1)
    pdf.cell(200, 10, txt=f"Age: {data['Age']}   Gender: {data['Gender']}", ln=1)
    pdf.cell(200, 10, txt=f"Pregnancies: {data['Pregnancies']}   DPF: {data['DPF']}", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Health Metrics", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Glucose: {data['Glucose']}     Insulin: {data['Insulin']}", ln=1)
    pdf.cell(200, 10, txt=f"Blood Pressure: {data['Blood Pressure']}     Skin Thickness: {data['Skin Thickness']}", ln=1)
    pdf.cell(200, 10, txt=f"Weight: {data['Weight (kg)']} kg", ln=1)
    pdf.cell(200, 10, txt=f"Height: {data['Height (cm)']} cm", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Derived Indicators", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=1)
    pdf.cell(200, 10, txt=f"Total Risk Score: {risk_score}", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Prediction Result", ln=1)
    pdf.set_font("Arial", '', 12)
    result_text = "Diabetic" if prediction == 1 else "Non-Diabetic"
    pdf.cell(200, 10, txt=f"Prediction: {result_text}", ln=1)

    filename = f"diabetes_prediction_{data['Name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

def download_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return f'<a href="data:application/pdf;base64,{base64_pdf}" download="{file_path}">📄 Download Your PDF Report</a>'

# Streamlit App
st.set_page_config(page_title="Personalized Diabetes Prediction", layout="centered")
st.title("🩺 Personalized Diabetes Prediction App")

user_name = st.text_input("Enter your name")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
skinthickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
height = st.number_input("Height (cm)", min_value=50.0, value=170.0)
insulin = st.number_input("Insulin Level", min_value=0.0, value=100.0)
bloodpressure = st.number_input("Blood Pressure", min_value=0.0, value=80.0)
dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, max_value=3.0, value=0.5, help="A score indicating the genetic likelihood of diabetes based on family history. Typical values range from 0.1 to 2.5.")
checklist = st.multiselect("Check any of the following symptoms:", ["Frequent urination", "Excessive thirst", "Blurred vision", "Fatigue", "Slow healing wounds"])

if st.button("Predict"):
    bmi = calculate_bmi(weight, height)
    risk_score = calculate_risk_score(glucose, bmi, age, pregnancies)

    input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bloodpressure,
        'SkinThickness': skinthickness,
        'BMI': bmi,
        'Age': age,
        'DPF_log': np.log1p(dpf),
        'Insulin_log': np.log1p(insulin),
        'BP_Deviation': abs(bloodpressure - 80),
        'Glucose_BMI': glucose * bmi,
        'Total_Risk_Score': risk_score,
        'is_obese': int(bmi >= 30),
        'is_high_glucose': int(glucose >= 140),
        'is_high_bp': int(bloodpressure >= 90),
        'is_high_insulin': int(insulin >= 200),
        'is_high_dpf': int(dpf >= 1.0)
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.warning("⚠️ This result suggests a higher risk of diabetes. Please consult a healthcare provider for further testing.")
        st.subheader("Recommended Diet Tips for Managing Diabetes")
        st.markdown("""
        - **Choose complex carbs** like whole grains, lentils, and vegetables instead of refined carbs.
        - **Eat more fiber**: Vegetables, beans, and oats help regulate blood sugar.
        - **Limit sugary drinks and processed foods**.
        - **Control portion sizes** and space meals evenly.
        - **Healthy fats**: Choose olive oil, nuts, and avocados in moderation.
        - **Stay hydrated** and reduce sodium intake.

        _Always consult with a registered dietitian for personalized guidance._
        """)
        st.subheader("Physical Activity Tips")
        st.markdown("""
        - Aim for at least **150 minutes of moderate activity** per week (e.g., brisk walking, swimming).
        - Incorporate **strength training** 2–3 times per week.
        - Try to **avoid sitting for long periods** — move every 30–60 minutes.
        - Stretch regularly and consider activities like **yoga or cycling** for endurance.
        """)
        st.subheader("Helpful Resources")
        st.markdown("""
        - [American Diabetes Association](https://www.diabetes.org/healthy-living)
        - [CDC Diabetes Prevention Program](https://www.cdc.gov/diabetes/prevention/index.html)
        - [WHO Diabetes Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)
        """)
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
