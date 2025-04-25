import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- Config ----------
st.set_page_config(page_title="Energy Prediction ⚡", layout="wide")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# ---------- Assets ----------
logo = Image.open("assets/logo.png")
cartoon = Image.open("assets/cartoon.png")

st.image(logo, width=100)
st.title("⚡ Energy Consumption Prediction Dashboard")
st.markdown("Monitor your energy usage, estimate carbon emissions, and help fight global warming 🌍.")

st.markdown("---")
input_mode = st.radio("Choose Input Method", ["📁 Upload CSV", "✍️ Manual Input"])

FEATURES = ['temperature', 'humidity', 'windspeed', 'hour', 'dayofweek', 'is_holiday']
EMISSION_FACTOR = 0.475  # kg CO₂ / kWh

# ---------- CSV Upload ----------
if input_mode == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload your feature CSV file", type=["csv"])
    
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Sample")
        st.dataframe(input_df.head())

        # Prediction
        predictions = model.predict(input_df)
        input_df["Predicted Energy Consumption (kWh)"] = predictions
        input_df["Estimated CO₂ Emissions (kg)"] = input_df["Predicted Energy Consumption (kWh)"] * EMISSION_FACTOR

        st.markdown("### 🔮 Prediction Results")
        st.dataframe(input_df)

        # Cartoon Result
        st.image(cartoon, caption="Let’s keep the planet cool 🌍💨", width=250)

        # Line Chart
        st.markdown("### 📈 Energy Forecast")
        fig = px.line(input_df, y="Predicted Energy Consumption (kWh)", title="Predicted Energy Usage")
        st.plotly_chart(fig, use_container_width=True)

        # Emission Chart
        st.markdown("### 🌫️ CO₂ Emission Chart")
        fig2 = px.bar(input_df, y="Estimated CO₂ Emissions (kg)", title="Estimated Carbon Emissions")
        st.plotly_chart(fig2, use_container_width=True)

        # Download CSV
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions with Emissions", csv, "energy_predictions.csv", "text/csv")

# ---------- Manual Input ----------
else:
    st.subheader("Enter Feature Values Manually")

    input_values = {}
    for feature in FEATURES:
        default = 0.0 if feature != "is_holiday" else 0
        val = st.number_input(f"{feature.capitalize()}", value=default)
        input_values[feature] = val

    input_df = pd.DataFrame([input_values])

    if st.button("Predict"):
        pred_energy = model.predict(input_df)[0]
        pred_co2 = pred_energy * EMISSION_FACTOR

        st.success(f"🔋 Predicted Energy Consumption: **{pred_energy:.2f} kWh**")
        st.info(f"🌫️ Estimated CO₂ Emissions: **{pred_co2:.2f} kg CO₂**")
        st.image(cartoon, caption="🌍 Be the energy change!", width=250)

# ---------- Footer ----------
st.markdown("---")
st.caption("💡 Project by Energy Forecasting Group 2025 | 🌍 Saving the planet, one watt at a time.")
