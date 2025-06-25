import streamlit as st
import numpy as np
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
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=1, align='C')
    pdf.ln(10)
    
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)

    pdf.ln(5)
    pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=1)
    pdf.cell(200, 10, txt=f"Total Risk Score: {risk_score}", ln=1)
    pdf.cell(200, 10, txt=f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}", ln=1)

    file_path = "diabetes_prediction_report.pdf"
    pdf.output(file_path)
    return file_path

# File download link
def download_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="diabetes_prediction_report.pdf">📥 Download Prediction Report</a>'
        return href

# Custom style
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton > button { background-color: #008080; color: white; font-weight: bold; }
    .stDownloadButton { color: #008080; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Personalized Diabetes Prediction")
st.subheader("Input your health information below:")

# User name input
user_name = st.text_input("👤 Enter your name:", value="")

# Sidebar with checklist
with st.sidebar:
    st.markdown("### Optional Info")
    checklist = st.multiselect("Check if you have these symptoms:",
        ["Frequent urination", "Excessive thirst", "Fatigue", "Blurred vision"])

# User inputs
age = st.slider("Age", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.slider("Glucose", min_value=50, max_value=200, value=110)
skinthickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
weight = st.slider("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.slider("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
insulin = st.slider("Insulin", min_value=0.0, max_value=600.0, value=100.0)
bloodpressure = st.slider("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)

# Predict button
if st.button("🔍 Predict"):
    bmi = calculate_bmi(weight, height)
    risk_score = calculate_risk_score(glucose, bmi, age, pregnancies)

    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, bmi, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    st.markdown("## 🧾 Prediction Result")
    st.write(f"**Patient Name:** {user_name if user_name else 'N/A'}")
    st.write("**Prediction:**", "🟥 Diabetic" if prediction == 1 else "🟩 Non-Diabetic")
    st.write("**BMI:**", f"{bmi}")
    st.write("**Total Risk Score:**", f"{risk_score}")
    st.write("**Confidence:**", f"{round(probability * 100, 2)}%")

    if prediction == 1:
        st.warning("⚠️ This result suggests a higher risk of diabetes. Please consult a healthcare provider for further testing.")
    else:
        st.success("✅ This result suggests a lower risk of diabetes.")

    # SHAP explanation
    st.subheader("🔍 Visual Explanation (SHAP)")
    explainer = shap.Explainer(model, X_train_resampled)
    shap_values = explainer(input_scaled)
    shap.initjs()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    st.pyplot(bbox_inches='tight')

    # Generate PDF
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
        "Symptoms Checked": ', '.join(checklist) if checklist else "None"
    }

    pdf_file = generate_pdf(user_info, prediction, bmi, risk_score)
    st.markdown(download_pdf(pdf_file), unsafe_allow_html=True)
