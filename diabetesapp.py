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
from datetime import datetime

def generate_pdf(data, prediction, bmi, risk_score):
    pdf = FPDF()
    pdf.add_page()
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated on: {report_date}", ln=1, align='R')
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
    pdf.cell(200, 10, txt="Health Metrics", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Glucose: {data['Glucose']}     Insulin: {data['Insulin']}", ln=1)
    pdf.cell(200, 10, txt=f"Blood Pressure: {data['Blood Pressure']}     Skin Thickness: {data['Skin Thickness']}", ln=1)
    pdf.cell(200, 10, txt=f"Weight: {data['Weight (kg)']} kg", ln=1)
    pdf.cell(200, 10, txt=f"Height: {data['Height (cm)']} cm", ln=1)

    # Derived indicators
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Derived Indicators", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=1)
    pdf.cell(200, 10, txt=f"Total Risk Score: {risk_score}", ln=1)

    # Prediction result
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Prediction Result", ln=1)
    pdf.set_font("Arial", '', 12)
    if prediction == 1:
        st.warning("⚠️ This result suggests a higher risk of diabetes. Please consult a healthcare provider for further testing.")

        # Recommended Diet
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

        # Recommended Exercise
        st.subheader("Physical Activity Tips")
        st.markdown("""
        - Aim for at least **150 minutes of moderate activity** per week (e.g., brisk walking, swimming).
        - Incorporate **strength training** 2–3 times per week.
        - Try to **avoid sitting for long periods** — move every 30–60 minutes.
        - Stretch regularly and consider activities like **yoga or cycling** for endurance.
        """)

        # External Resources
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
