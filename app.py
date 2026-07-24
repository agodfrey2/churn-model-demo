import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    with open("churn_xgb_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_xgb_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🥗 Healthy Meals Renewal Predictor")
st.caption("Predict the likelihood that a Healthy Meals subscriber will renew their subscription.")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

col1, col2 = st.columns(2)
with col1: 
    st.subheader("Customer Information")
    age  = st.number_input("Age", min_value=18, max_value=100, value=35)
    income_level = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
    education = st.radio("Education",     ["High School", "Graduate","Post-Graduate", "Other"])
    device_type = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])
    tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=5, value=5)

with col2: 
    st.subheader("Activity Metrics")
    total_sessions = st.number_input("Total Sessions", min_value=0, value=50)
    gross_session_length = st.number_input("Gross Session Length (Minutes)",min_value=0,value=1000, step=1)
    active_days = st.number_input("Active Days", min_value=0, value=10)
    active_quarters = st.number_input("Active Quarters", min_value=0, value=2)
    avg_sessions_per_active_quarter = st.number_input("Avg Sessions per Active Quarter", min_value=0,value=25)
    avg_session_length = st.number_input("Avg Session Length", min_value=0, value=30)
    sessions_per_active_day = st.number_input("Sessions per Active Day", min_value=0, value=5)
    days_since_last_activity = st.number_input("Days Since Last Activity", min_value=0, value=30)

button_col1, button_col2, button_col3 = st.columns([1,1,1])

with button_col2:
    predict = st.button("Predict")

if predict:

    # Build categorical DataFrame — column names must match encoder exactly
    raw = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features first, then encoded dummies — must match training column order
    numeric_df = pd.DataFrame([{
    'TOTAL_SESSIONS': total_sessions,
    'GROSS_SESSION_LENGTH': gross_session_length,
    'ACTIVE_DAYS': active_days,
    'ACTIVE_QUARTERS': active_quarters,
    'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_active_quarter,
    'AVG_SESSION_LENGTH': avg_session_length,
    'SESSIONS_PER_ACTIVE_DAY': sessions_per_active_day,
    'DAYS_SINCE_LAST_ACTIVITY': days_since_last_activity,
    'AGE': age,
    'TECH_COMFORT_SCORE': tech_comfort_score
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    
    #Debug
    #st.write(input_df.columns.tolist())

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.divider()
    
    st.metric("Renewal Probability", f"{probability:.1%}")
    st.caption(f"Predicted renewal Likelihood: {probability:.1%}")

    st.progress(int(probability*100))
    
    if risk == "High":
        st.error(f"🚨Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"⚠️Churn Risk: {risk}")
    else:
        st.success(f"✅Churn Risk: {risk}")




