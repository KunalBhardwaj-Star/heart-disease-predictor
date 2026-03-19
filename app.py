import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("heart_model.pkl")

# -----------------------------
# PDF Function
# -----------------------------
def create_pdf(result_text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Heart Disease Prediction Report", styles["Title"]))
    content.append(Paragraph(result_text, styles["Normal"]))

    doc.build(content)

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("### Enter real patient details below")

# -----------------------------
# 🩺 Vital Signs
# -----------------------------
st.subheader("🩺 Vital Signs")

bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
chol = st.number_input("Cholesterol Level (mg/dL)", 100, 400, 200)
BMI = st.number_input("BMI", 10.0, 50.0, 25.0)

HighBP = 1 if bp >= 130 else 0
HighChol = 1 if chol >= 240 else 0
CholCheck = 1

# -----------------------------
# 🧬 Lifestyle
# -----------------------------
st.subheader("🧬 Lifestyle")

smoker = st.radio("Do you smoke?", ["No", "Yes"])
Smoker = 1 if smoker == "Yes" else 0

activity = st.radio("Do you exercise regularly?", ["Yes", "No"])
PhysActivity = 1 if activity == "Yes" else 0

fruits = st.radio("Do you eat fruits daily?", ["Yes", "No"])
Fruits = 1 if fruits == "Yes" else 0

veggies = st.radio("Do you eat vegetables daily?", ["Yes", "No"])
Veggies = 1 if veggies == "Yes" else 0

alcohol = st.radio("Heavy alcohol consumption?", ["No", "Yes"])
HvyAlcoholConsump = 1 if alcohol == "Yes" else 0

# -----------------------------
# 📊 Health History
# -----------------------------
st.subheader("📊 Health History")

stroke = st.radio("History of Stroke?", ["No", "Yes"])
Stroke = 1 if stroke == "Yes" else 0

diabetes = st.selectbox("Diabetes Level", [0, 1, 2])

healthcare = st.radio("Access to Healthcare?", ["Yes", "No"])
AnyHealthcare = 1 if healthcare == "Yes" else 0

nodoc = st.radio("Avoid doctor due to cost?", ["No", "Yes"])
NoDocbcCost = 1 if nodoc == "Yes" else 0

genhlth = st.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)

menthlth = st.slider("Mental Health (days in last month)", 0, 30, 5)
physhlth = st.slider("Physical Health (days in last month)", 0, 30, 5)

diffwalk = st.radio("Difficulty Walking?", ["No", "Yes"])
DiffWalk = 1 if diffwalk == "Yes" else 0

# -----------------------------
# 👤 Personal Info
# -----------------------------
st.subheader("👤 Personal Info")

sex = st.radio("Sex", ["Female", "Male"])
Sex = 1 if sex == "Male" else 0

age = st.slider("Age Category (1-13)", 1, 13, 5)
education = st.selectbox("Education Level (1-6)", [1,2,3,4,5,6])
income = st.selectbox("Income Level (1-8)", [1,2,3,4,5,6,7,8])

# -----------------------------
# 🔮 Prediction
# -----------------------------
st.markdown("---")

if st.button("Predict"):

    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                            diabetes, PhysActivity, Fruits, Veggies,
                            HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
                            genhlth, menthlth, physhlth, DiffWalk,
                            Sex, age, education, income]])

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
    else:
        prob = None

    st.subheader("🩺 Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        result_text = "High Risk of Heart Disease"
    else:
        st.success("✅ Low Risk of Heart Disease")
        result_text = "Low Risk of Heart Disease"

    # -----------------------------
    # 📊 Risk Graph
    # -----------------------------
    if prob is not None:
        st.subheader("📊 Risk Visualization")

        fig, ax = plt.subplots()
        labels = ["Low Risk", "High Risk"]
        values = [1 - prob, prob]

        ax.bar(labels, values)
        ax.set_ylabel("Probability")

        st.pyplot(fig)

    # -----------------------------
    # 🧠 SHAP (FIXED)
    # -----------------------------
    st.subheader("🧠 Model Explanation")

    try:
        explainer = shap.LinearExplainer(model, input_data)
        shap_values = explainer.shap_values(input_data)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP not supported for this model: {e}")

    # -----------------------------
    # 📄 PDF Report
    # -----------------------------
    create_pdf(result_text)

    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f, file_name="report.pdf")

# -----------------------------
# 🎨 Styling
# -----------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)