import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="PaviRaju/tourism-wellness-package-purchase", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Welness Package Customer Prediction App")
st.write("The Customer Churn Prediction App is an internal tool for bank staff that predicts whether customers are at risk of churning based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

# Numerical features
age = st.number_input("Age", min_value=0, max_value=120, step=1)
city_tier = st.number_input("City Tier", min_value=1, max_value=3, step=1)
num_person_visiting = st.number_input("Number Of Person Visiting", min_value=0, step=1)
num_followups = st.number_input("Number Of Followups", min_value=0, step=1)
duration_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, step=1)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited", "Other"])
occupation = st.selectbox("Occupation", ["Salaried", "Business", "Student", "Other"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Other"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Premium", "Other"])
preferred_property_star = st.selectbox("Preferred Property Star", ["3 Star", "4 Star", "5 Star"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch,
    "TypeofContact": typeof_contact,
    "Occupation": occupation,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Designation": designation,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Take Product" if prediction == 1 else "Not take Product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
