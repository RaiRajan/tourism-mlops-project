"""Streamlit front end for the Wellness Tourism Package purchase predictor."""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("Visit with Us — Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details to estimate the likelihood that they will purchase "
    "the Wellness Tourism Package. This allows the sales team to prioritise "
    "high-potential customers before making contact."
)


@st.cache_resource
def load_model():
    """Load the model committed to the repository by the GitHub Actions pipeline."""
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Trained model not found. Run the GitHub Actions pipeline first.")
    st.stop()

st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    number_of_trips = st.number_input("Average Trips per Year", 0, 30, 3)

with col2:
    passport = st.selectbox("Holds a Passport", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car", ["Yes", "No"])
    number_of_children_visiting = st.number_input("Children Below Age 5 Visiting", 0, 5, 1)
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 22000, step=500)
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 200, 15)

threshold = st.slider(
    "Decision threshold", 0.05, 0.95, 0.50, 0.05,
    help="Lower the threshold to reach more potential buyers at the cost of "
         "contacting more customers who will not convert.",
)

# Collect all user inputs into a single-row dataframe matching the training schema
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("View the dataframe passed to the model"):
    st.dataframe(input_df)

if st.button("Predict", type="primary"):
    proba = float(model.predict_proba(input_df)[0, 1])
    label = int(proba >= threshold)
    st.metric("Purchase probability", f"{proba:.1%}")
    if label:
        st.success("**Likely to purchase** — prioritise this customer for outreach.")
    else:
        st.info("**Unlikely to purchase** — deprioritise or offer an alternative package.")
    st.caption(f"Decision threshold applied: {threshold:.2f}")
