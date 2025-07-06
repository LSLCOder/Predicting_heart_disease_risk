import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import gdown
import time

# ----------------------------
# Load model from Google Drive
# ----------------------------
st.set_page_config(page_title="Heart Disease Risk Checker", layout="centered")
model_path = "heart_disease_model.pkl"
file_id = "1_OqUNI5f3q_BgvjtNnepRIebqgg_6VHR"
url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not present
if not os.path.exists(model_path):
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("### 📥 Downloading Model from Google Drive...")
        progress_bar = st.progress(0, text="Initializing...")

        for i in range(0, 80, 10):
            time.sleep(0.1)
            progress_bar.progress(i, text=f"Downloading... {i}%")

        gdown.download(url, model_path, quiet=True)

        for i in range(80, 101, 5):
            time.sleep(0.05)
            progress_bar.progress(i, text=f"Finalizing... {i}%")

        st.success("✅ Model downloaded successfully!")
        st.markdown("</div>", unsafe_allow_html=True)
        time.sleep(1.5)
        placeholder.empty()

# Load model
model = joblib.load(model_path)

# ----------------------------
# App Layout
# ----------------------------
st.markdown("""
<div style="background-color:#f5f7fa;padding:25px 30px;border-radius:15px;
box-shadow:0 0 6px rgba(0,0,0,0.05);margin-bottom:30px;">
<h1 style='color:#1f77b4;'>🫀 Heart Disease Risk Checker</h1>
<p style='font-size: 16px; color: #555;'>Please fill out the patient’s clinical and lifestyle details.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("📊 Model Info"):
    st.markdown("**Random Forest Classifier** trained on 17 health features.")
    st.markdown("Balanced using SMOTE, Accuracy ~90%, Recall ~92%")

# ----------------------------
# Helper
# ----------------------------
def encode_binary(option): return 1 if option else 0

# ----------------------------
# Inputs
# ----------------------------
st.header("👤 Patient Information")
col1, col2 = st.columns([1, 1])
with col1:
    sex = st.radio("Sex", ["Male", "Female"])
    age_cat = st.selectbox("Age Group", ["18-24", "25-29", "30-34", "35-39",
                                         "40-44", "45-49", "50-54", "55-59",
                                         "60-64", "65-69", "70-74", "75-79", "80 or older"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    race = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"])
with col2:
    gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
    sleep_time = st.slider("Sleep Time (hrs/night)", 0, 24, 7)

st.markdown("---")

st.header("🏥 Medical History")
col3, col4 = st.columns([1, 1])
with col3:
    diabetic = st.selectbox("Diabetes Status", ["No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"])
    stroke = st.checkbox("History of Stroke")
    asthma = st.checkbox("Has Asthma")
with col4:
    kidney_disease = st.checkbox("Has Kidney Disease")
    skin_cancer = st.checkbox("Has Skin Cancer")
    diff_walking = st.checkbox("Difficulty Walking")

st.markdown("---")

st.header("🏃 Lifestyle & Mental Health")
col5, col6 = st.columns([1, 1])
with col5:
    smoking = st.checkbox("Currently Smokes")
    alcohol = st.checkbox("Drinks Alcohol")
    physical_activity = st.checkbox("Physically Active")
with col6:
    physical_health = st.slider("Days Physical Health Not Good", 0, 30, 5)
    mental_health = st.slider("Days Mental Health Not Good", 0, 30, 5)

# ----------------------------
# Encoding
# ----------------------------
sex = encode_binary(sex == "Male")
smoking = encode_binary(smoking)
alcohol = encode_binary(alcohol)
stroke = encode_binary(stroke)
diff_walking = encode_binary(diff_walking)
physical_activity = encode_binary(physical_activity)
asthma = encode_binary(asthma)
kidney_disease = encode_binary(kidney_disease)
skin_cancer = encode_binary(skin_cancer)

diabetic_map = {
    "No": 0, "No, borderline diabetes": 1, "Yes (during pregnancy)": 2, "Yes": 3
}
diabetic = diabetic_map[diabetic]

age_map = {
    "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3,
    "40-44": 4, "45-49": 5, "50-54": 6, "55-59": 7,
    "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80 or older": 12
}
age_cat = age_map[age_cat]

gen_health_map = {
    "Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4
}
gen_health = gen_health_map[gen_health]

race_map = {
    "White": 0, "Black": 1, "Asian": 2,
    "American Indian/Alaskan Native": 3, "Other": 4, "Hispanic": 5
}
race = race_map[race]

# ----------------------------
# Feature Array with Column Names
# ----------------------------
feature_names = [
    "BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth",
    "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic",
    "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer"
]

features = pd.DataFrame([[
    bmi, smoking, alcohol, stroke, physical_health, mental_health,
    diff_walking, sex, age_cat, race, diabetic, physical_activity,
    gen_health, sleep_time, asthma, kidney_disease, skin_cancer
]], columns=feature_names)

# ----------------------------
# Predict
# ----------------------------
if st.button("🩺 Predict Heart Disease Risk"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ The patient is **AT RISK**.\nConfidence: {probability:.2%}")
    else:
        st.success(f"✅ The patient is **NOT at risk**.\nConfidence: {1 - probability:.2%}")
