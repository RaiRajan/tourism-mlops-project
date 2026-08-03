"""Streamlit front end for the Wellness Tourism Package purchase predictor."""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("Visit with Us — Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details to estimate the likelihood that they will purchase the "
    "Wellness Tourism Package, so the sales team can prioritise high-potential customers "
    "before making contact."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer details")
    age = st.number_input("Age", 18, 100, 37)
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City tier", [1, 2, 3], index=2)
    occupation = st.selectbox("Occupation",
                              ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status",
                                  ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation",
                               ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", 1000, 100000, 17000, step=500)
    passport = st.selectbox("Holds a passport", ["Yes", "No"])
    own_car = st.selectbox("Owns a car", ["Yes", "No"], index=1)

with col2:
    st.subheader("Trip preferences and pitch")
    persons = st.number_input("Number of persons visiting", 1, 10, 3)
    children = st.number_input("Number of children visiting", 0, 5, 1)
    trips = st.number_input("Average number of trips per year", 0, 30, 4)
    property_star = st.selectbox("Preferred property star", [3, 4, 5])
    product_pitched = st.selectbox("Product pitched",
                                   ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of pitch (minutes)", 1, 180, 11)
    followups = st.number_input("Number of follow-ups", 0, 10, 4)
    satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=4)

threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

row = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Model input"):
    st.dataframe(row)

if st.button("Predict"):
    prob = float(model.predict_proba(row)[:, 1][0])
    st.metric("Purchase probability", f"{prob:.1%}")
    if prob >= threshold:
        st.success("Likely to purchase — prioritise this customer for outreach.")
    else:
        st.info("Unlikely to purchase — deprioritise or nurture with lower-cost channels.")
    st.caption(f"Classified at a threshold of {threshold:.2f}. "
               "Lower the threshold to catch more buyers at the cost of more wasted calls.")
