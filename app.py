import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.title("🫀 Heart Disease Predictor")
st.write("Fill in the patient details below and click **Predict**.")

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina",
                                3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    bp = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("FBS over 120", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ekg = st.selectbox("EKG Results", options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])

with col2:
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Angina", options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0,
        value=0.0, step=0.1)
    slope_st = st.selectbox("Slope of ST", options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    vessels_fluro = st.selectbox("Vessels Fluro", options=[0, 1, 2, 3])
    thallium = st.selectbox("Thallium", options=[3, 6, 7],
        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

st.divider()

if st.button("Predict", use_container_width=True, type="primary"):

    input_data = np.array([[
        age, sex, chest_pain, bp, cholesterol, fbs,
        ekg, max_hr, exercise_angina, st_depression,
        slope_st, vessels_fluro, thallium
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Result
    if prediction == 1:
        st.error(f"⚠️ **Heart Disease: Presence** — Confidence: {probability[1]*100:.1f}%")
    else:
        st.success(f"✅ **Heart Disease: Absence** — Confidence: {probability[0]*100:.1f}%")

    # SHAP Feature Importance
    st.subheader("Feature Importance")
    st.caption("Shows which features pushed the prediction — red increases risk, green decreases it.")

    feature_names = ["Age", "Sex", "Chest Pain Type", "BP", "Cholesterol",
                     "FBS over 120", "EKG Results", "Max HR", "Exercise Angina",
                     "ST Depression", "Slope of ST", "Vessels Fluro", "Thallium"]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_data)
    shap_values = explanation.values

    vals = shap_values[0][1:]        
    names = feature_names[1:]        

    sorted_idx = np.argsort(np.abs(vals))
    sorted_vals = vals[sorted_idx]
    sorted_names = np.array(names)[sorted_idx]
    colors = ["#e05c5c" if v > 0 else "#3ecf6e" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(sorted_names, sorted_vals, color=colors, height=0.6)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel("SHAP Value  (red = increases risk · green = decreases risk)", fontsize=8)
    ax.set_title("What drove this prediction?", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
