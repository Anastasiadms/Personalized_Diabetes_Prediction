import streamlit as st
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
import base64
from datetime import datetime

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
    filename = f"diabetes_prediction_{data['Name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated on: {report_date}", ln=1, align='R')
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
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
    if prediction == 1:
        pdf.set_text_color(255, 0, 0)
    else:
        pdf.set_text_color(0, 128, 0)
    result_text = "Diabetic" if prediction == 1 else "Non-Diabetic"
    pdf.cell(200, 10, txt=f"Prediction: {result_text}", ln=1)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Symptoms Checked", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=data['Symptoms Checked'])

    pdf.output(filename)
    return filename

def download_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{file_path}">📥 Download Prediction Report</a>'
        return href

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("Diabetes Prediction")

user_name = st.text_input("👤 Enter your name:", value="")
age = st.slider("Age", 1, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
pregnancies = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 50, 200, 110)
skinthickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
weight = st.number_input("Weight (kg)", 30.0, 200.0, step=0.1)
height = st.number_input("Height (cm)", 100.0, 220.0, step=0.1)
insulin = st.slider("Insulin", 0.0, 600.0, 100.0)
dpf = st.number_input("Diabetes Pedigree Function (DPF)", 0.0, 3.0, value=0.5, help="A score indicating the genetic likelihood of diabetes based on family history. Typical values range from 0.1 to 2.5.")
bloodpressure = st.slider("Blood Pressure", 0.0, 200.0, 80.0)

st.markdown("### 🤒 Symptoms Checklist (Optional)")
checklist = st.multiselect(
    "Do you experience any of the following symptoms?",
    ["Frequent urination", "Excessive thirst", "Fatigue", "Blurred vision"]
)

if st.button("🔍 Predict"):
    bmi = calculate_bmi(weight, height)
    risk_score = calculate_risk_score(glucose, bmi, age, pregnancies)
    glucose_bmi = glucose * bmi
    insulin_log = np.log1p(insulin)
    dpf_log = np.log1p(dpf)
    bp_deviation = abs(bloodpressure - 80)

    age_groups = {'AgeGroup_31-40': 0, 'AgeGroup_41-50': 0, 'AgeGroup_51-60': 0, 'AgeGroup_60+': 0}
    if 31 <= age <= 40: age_groups['AgeGroup_31-40'] = 1
    elif 41 <= age <= 50: age_groups['AgeGroup_41-50'] = 1
    elif 51 <= age <= 60: age_groups['AgeGroup_51-60'] = 1
    elif age > 60: age_groups['AgeGroup_60+'] = 1

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
    st.write("**BMI:**", bmi, help="Body Mass Index — a measure of body fat based on height and weight.")
    st.write("**Total Risk Score:**", risk_score, help="A custom score combining glucose, BMI, age, and pregnancies to estimate diabetes risk.")
    st.write("**Confidence:**", f"{round(probability * 100, 2)}%")

    if prediction == 1:
        st.warning("⚠️ This result suggests a higher risk of diabetes. Please consult a healthcare provider for further testing.")

        st.subheader("🥗 Recommended Diet Tips for Managing Diabetes")
        st.markdown("""
        - **Choose complex carbs** like whole grains, lentils, and vegetables instead of refined carbs.
        - **Eat more fiber**: Vegetables, beans, and oats help regulate blood sugar.
        - **Limit sugary drinks and processed foods**.
        - **Control portion sizes** and space meals evenly.
        - **Healthy fats**: Choose olive oil, nuts, and avocados in moderation.
        - **Stay hydrated** and reduce sodium intake.

        _Always consult with a registered dietitian for personalized guidance._
        """)

        st.subheader("🏃‍♂️ Physical Activity Tips")
        st.markdown("""
        - Aim for at least **150 minutes of moderate activity** per week (e.g., brisk walking, swimming).
        - Incorporate **strength training** 2–3 times per week.
        - Try to **avoid sitting for long periods** — move every 30–60 minutes.
        - Stretch regularly and consider activities like **yoga or cycling** for endurance.
        """)

        st.subheader("🔗 Helpful Resources")
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

