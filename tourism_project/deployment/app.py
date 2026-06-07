import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="PaviRaju/tourism-wellness-package-purchase", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Wellness Package Customer Prediction App")
st.write("The Tourism Wellness Package Customer App is an internal tool for travel company that predicts whether customers will buy a Wellness package.")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Numerical features
age = st.slider("Age", 18, 70, 30)
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
number_of_person_visiting = st.slider("Number Of Person Visiting", 1, 5, 2)
number_of_children_visiting = st.slider("Number Of Children Visiting", 0, 5, 0)
duration_of_pitch = st.slider("Duration Of Pitch (minutes)", 0, 60, 10)
city_tier = st.selectbox("City Tier", [1, 2, 3])
gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male']) # 'Fe Male' is a common typo in datasets
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
number_of_trips = st.slider("NumberOfTrips (annually)", 0, 20, 5)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
number_of_followups = st.slider("NumberOfFollowups", 0, 10, 3)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Other"])
monthly_income = st.slider("Monthly Income", 0, 100000, 25000)

# Convert categorical inputs to match model training
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

# Predict button
if st.button("Predict"):
    prediction= model.predict(input_data)[0]
    if prediction == 1:
            st.success(f"Prediction: This customer is LIKELY to purchase the package")
    else:
            st.info(f"Prediction: This customer is UNLIKELY to purchase the package")
