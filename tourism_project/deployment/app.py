import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Download and load the model
model_path = hf_hub_download(repo_id="PaviRaju/tourism-wellness-package-purchase", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🌍 Tourism Wellness Package Prediction App")
st.write("This internal tool helps predict whether a customer is likely to purchase a Wellness Tourism Package based on their profile.")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Other"])

with col2:
    type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    number_of_person_visiting = st.slider("Number Of Person Visiting", 1, 5, 2)
    number_of_children_visiting = st.slider("Number Of Children Visiting", 0, 5, 0)
    duration_of_pitch = st.slider("Duration Of Pitch (minutes)", 0, 60, 10)
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    monthly_income = st.slider("Monthly Income", 0, 100000, 25000)

# Additional inputs
number_of_trips = st.slider("Number Of Trips (annually)", 0, 20, 5)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
number_of_followups = st.slider("Number Of Followups", 0, 10, 3)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

# Prepare input data
input_data = pd.DataFrame([{
    'Unnamed: 0': 0,
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown(f"<div class='prediction-box' style='background-color:#d4edda; color:#155724;'>✅ This customer is LIKELY to purchase the package<br>Confidence: {prob:.2%}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box' style='background-color:#f8d7da; color:#721c24;'>❌ This customer is UNLIKELY to purchase the package<br>Confidence: {prob:.2%}</div>", unsafe_allow_html=True)
