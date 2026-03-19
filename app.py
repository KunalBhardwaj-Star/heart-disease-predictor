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
# Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# -----------------------------
# 🎨 Custom Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #eef2f3, #ffffff);
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    border: none;
    padding: 10px;
}

h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: #ff4b4b;'>
❤️ Heart Disease Prediction Dashboard
</h1>
""", unsafe_allow_html=True)

st.caption("⚠️ This is not a medical diagnosis. Consult a doctor.")

# -----------------------------
# 🩺 Vital Signs
# -----------------------------
st.subheader("🩺 Vital Signs")

col1, col2, col3 = st.columns(3)

with col1:
    bp = st.number_input("Blood Pressure", 80, 200, 120)
with col2:
    chol = st.number_input("Cholesterol", 100, 400, 200)
with col3:
    BMI = st.number_input("BMI", 10.0, 50.0, 25.0)

HighBP = 1 if bp >= 130 else 0
HighChol = 1 if chol >= 240 else 0
CholCheck = 1

# -----------------------------
# 🧬 Lifestyle
# -----------------------------
st.subheader("🧬 Lifestyle")

col1, col2 = st.columns(2)

with col1:
    smoker = st.radio("Smoking", ["No", "Yes"])
    fruits = st.radio("Fruits Intake", ["No", "Yes"])
    alcohol = st.radio("Alcohol", ["No", "Yes"])

with col2:
    activity = st.radio("Exercise", ["Yes", "No"])
    veggies = st.radio("Vegetables", ["Yes", "No"])

Smoker = 1 if smoker == "Yes" else 0
PhysActivity = 1 if activity == "Yes" else 0
Fruits = 1 if fruits == "Yes" else 0
Veggies = 1 if veggies == "Yes" else 0
HvyAlcoholConsump = 1 if alcohol == "Yes" else 0

# -----------------------------
# 📊 Health History
# -----------------------------
st.subheader("📊 Health History")

stroke = st.radio("Stroke History", ["No", "Yes"])
Stroke = 1 if stroke == "Yes" else 0

diabetes = st.selectbox("Diabetes Level", [0, 1, 2])

healthcare = st.radio("Healthcare Access", ["Yes", "No"])
AnyHealthcare = 1 if healthcare == "Yes" else 0

nodoc = st.radio("Avoid Doctor (Cost)", ["No", "Yes"])
NoDocbcCost = 1 if nodoc == "Yes" else 0

genhlth = st.slider("General Health", 1, 5, 3)
menthlth = st.slider("Mental Health Days", 0, 30, 5)
physhlth = st.slider("Physical Health Days", 0, 30, 5)

diffwalk = st.radio("Difficulty Walking", ["No", "Yes"])
DiffWalk = 1 if diffwalk == "Yes" else 0

# -----------------------------
# 👤 Personal Info
# -----------------------------
st.subheader("👤 Personal Info")

sex = st.radio("Sex", ["Female", "Male"])
Sex = 1 if sex == "Male" else 0

age = st.slider("Age Category", 1, 13, 5)
education = st.selectbox("Education Level", [1,2,3,4,5,6])
income = st.selectbox("Income Level", [1,2,3,4,5,6,7,8])

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

    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    # -----------------------------
    # 🩺 Result Card
    # -----------------------------
    st.markdown("## 🩺 Prediction Result")

    if prediction == 1:
        result_text = "High Risk of Heart Disease"
        st.markdown("""
        <div class="card">
            <h2 style='color:red;'>⚠️ High Risk</h2>
            <p>Patient shows higher probability of heart disease.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result_text = "Low Risk of Heart Disease"
        st.markdown("""
        <div class="card">
            <h2 style='color:green;'>✅ Low Risk</h2>
            <p>Patient shows lower probability of heart disease.</p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # 📊 Graph
    # -----------------------------
    if prob is not None:
        st.subheader("📊 Risk Visualization")

        fig, ax = plt.subplots()
        labels = ["Low Risk", "High Risk"]
        values = [1 - prob, prob]

        ax.bar(labels, values)
        ax.set_title("Heart Disease Risk Probability")

        st.pyplot(fig)

    # -----------------------------
    # 🧠 SHAP (Top Features Only)
    # -----------------------------
    st.subheader("🧠 Key Risk Factors")

    try:
        explainer = shap.LinearExplainer(model, input_data)
        shap_values = explainer.shap_values(input_data)

        impact = shap_values[0]
        top_indices = np.argsort(np.abs(impact))[-3:]

        for i in top_indices:
            st.write(f"• Feature {i} impact: {impact[i]:.3f}")

    except:
        st.warning("Explanation not available")

    # -----------------------------
    # 📄 PDF Report
    # -----------------------------
    create_pdf(result_text)

    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f, file_name="report.pdf")
