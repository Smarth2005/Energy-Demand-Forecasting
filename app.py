import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- Config ----------  
st.set_page_config(page_title="⚡ Energy Prediction Dashboard", layout="wide")

# ---------- Top Row with Logo and Cartoon in One Line ----------
logo = Image.open("assets/logo.png")
cartoon = Image.open("assets/cartoon.png")

# Create a layout with 3 columns (centered)
col1, col2, col3 = st.columns([1, 2, 1])

# Show cartoon and logo in a single line with the same width
with col2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(cartoon, caption="Let’s keep the planet cool 🌍💨", width=300) 
    with col2:
        st.image(logo, width=350) 

# ---------- Beautiful Quote Center Aligned ----------
st.markdown(
    """
    <h3 style="text-align: center; color: #2e8b57;">
    "The patterns of energy consumption are always changing-with every degree of temperature, every hour of the day. 
	Accurate prediction is not just about numbers, but about understanding the pulse of our environment."
    </h3>
    <p style="text-align: center;">Adapted from E3S Conference on Energy Consumption Prediction</p>
    """, 
    unsafe_allow_html=True
)

import streamlit as st

# Custom CSS for scrolling text
st.markdown(
    """
    <style>
    .scrolling-text {
        width: 100%;
        overflow: hidden;
        position: relative;
        background-color: #2e8b57;
        padding: 20px 0; /* Increased height */
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 48px; /* Optional: explicit height for consistency */
    }

    .scrolling-text span {
        font-size: 1.5rem !important;
        font-weight: bold;
        color: #fff; /* White text */
        white-space: nowrap;
        position: absolute;
        animation: scroll-left 15s linear infinite;
        left: 100%;
        text-shadow: 1px 1px 4px #000;
        margin: 0;
        padding-left: 10px;
        top: 50%;
        transform: translateY(-50%);
    }

    @keyframes scroll-left {
        0% {
            left: 100%;
        }
        100% {
            left: -100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use <span> instead of <p> to avoid Streamlit markdown quirks
st.markdown(
    """
    <div class="scrolling-text">
        <span>⚡ Track your energy consumption to save money and protect the planet. Every kWh matters! 🌍</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------- Load Model ----------
#@st.cache_resource
def load_model():
    import os
    import joblib
    
    model_path = "models/xgboost_model.pkl"
    return joblib.load(model_path)

# Load the model
model = load_model()

import streamlit as st
import streamlit.components.v1 as components

# Custom CSS for font, colors, layout, and radio label
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif !important;
    }
    .main-heading {
        font-size: 1.5rem;
        color: #2e8b57;
        font-weight: 600;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    .sub-heading {
        font-size: 1.15rem;
        color: #333;
        font-weight: 500;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }
    .engagement-section {
        background: #f1f8f6;
        border-radius: 10px;
        padding: 18px 24px 8px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(46,139,87,0.08);
    }
    .engagement-title {
        font-size: 1.1rem;
        color: #2e8b57;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .engagement-desc {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.2rem;
    }
    /* Make radio label bigger and green */
    div[data-testid="stRadio"] > label {
        font-size: 1.2rem !important;
        color: #2e8b57 !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Heading (smaller than before, but still prominent)
st.markdown(
    '<div class="main-heading">Monitor your energy usage, estimate carbon emissions, and help fight global warming 🌍.</div>',
    unsafe_allow_html=True
)

# Engagement Section
st.markdown(
    """
    <div class="engagement-section">
        <div class="engagement-title">✨ Engage With Your Data</div>
        <div class="engagement-desc">
            We encourage you to <b>input your own energy data</b> to visualize predictions and insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Mode Radio Button (label font size now increased)
input_mode = st.radio("Choose Input Method", ["📁 Upload CSV", "✍️ Manual Input"], horizontal=True)


FEATURES = ['temperature', 'humidity', 'windspeed', 'hour', 'dayofweek', 'is_holiday']
EMISSION_FACTOR = 0.475  # kg CO₂ / kWh

# ---------- CSV Upload ----------
uploaded_file = None
if input_mode == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload your feature CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # ✅ Save datetime separately for plotting
    if "datetime" in input_df.columns:
        datetime_col = pd.to_datetime(input_df["datetime"])
        input_df = input_df.drop(columns=["datetime"])  # ❌ Remove datetime for prediction
    else:
        st.warning("No datetime column found in the uploaded file.")

    st.write("📊 Uploaded Data Sample")
    st.dataframe(input_df.head())

    # 🔮 Prediction
    predictions = model.predict(input_df)
    input_df["Predicted Energy Consumption (kWh)"] = predictions
    input_df["Estimated CO₂ Emissions (kg)"] = input_df["Predicted Energy Consumption (kWh)"] * EMISSION_FACTOR

    # ✅ Add datetime back for plotting
    if "datetime_col" in locals():
        input_df["datetime"] = datetime_col
        input_df = input_df.sort_values("datetime")

    # 📈 Energy Forecast
    if "datetime" in input_df.columns:
        st.markdown("### 📈 Energy Forecast Over Time")
        fig = px.line(input_df, x="datetime", y="Predicted Energy Consumption (kWh)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🌫️ CO₂ Emission Chart")
        fig2 = px.bar(input_df, x="datetime", y="Estimated CO₂ Emissions (kg)")
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly Energy Consumption Comparison
    if "datetime" in input_df.columns:
        input_df['month'] = input_df['datetime'].dt.month
        monthly_consumption = input_df.groupby('month')['Predicted Energy Consumption (kWh)'].sum().reset_index()
        st.markdown("### 📊 Monthly Energy Consumption")
        fig_monthly = px.bar(monthly_consumption, x="month", y="Predicted Energy Consumption (kWh)", title="Monthly Energy Consumption")
        st.plotly_chart(fig_monthly, use_container_width=True)

    # Energy Savings Prediction (A placeholder prediction for future savings)
    if "Predicted Energy Consumption (kWh)" in input_df.columns:
        savings_pred = input_df["Predicted Energy Consumption (kWh)"].sum() * 0.1  # Example: 10% savings prediction
        st.markdown("### 💡 Predicted Energy Savings")
        st.write(f"By optimizing your energy usage, you can save approximately **{savings_pred:.2f} kWh** in the coming period.")

    # 📥 Download predictions
    csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions with Emissions", csv, "energy_predictions.csv", "text/csv")

    # Suggestions to Reduce Carbon Emissions
    st.markdown("### 🌱 Suggestions to Reduce Carbon Emissions")
    st.markdown(
    """
    - **Optimize HVAC systems**: Proper maintenance and settings can reduce energy consumption.
    - **Switch to LED lights**: LED lighting is more energy-efficient than conventional bulbs.
    - **Use energy-efficient appliances**: Choose appliances with a good energy rating.
    - **Consider renewable energy sources**: Solar and wind energy are great alternatives to fossil fuels.
    """
)
if input_mode == "✍️ Manual Input":
    st.markdown("### ✍️ Enter Feature Values Manually")

    from datetime import datetime
    import numpy as np
    import pandas as pd
    import pytz

    # Set timezone (for example, 'Asia/Kolkata')
    tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(tz).time()

    # === Input Fields ===
    date_input = st.date_input("Select Date", datetime.now().date())
    time_input = st.time_input("Select Time", current_time)
    datetime_input = datetime.combine(date_input, time_input)
    temp = st.number_input("Temperature (°C)", value=25.0)
    dew = st.number_input("Dew Point (°C)", value=10.0)
    humidity = st.slider("Humidity (%)", 0, 100, value=60)
    windgust = st.number_input("Wind Gust (km/h)", value=20.0)
    windspeed = st.number_input("Wind Speed (km/h)", value=10.0)
    sealevelpressure = st.number_input("Sea Level Pressure (hPa)", value=1013.0)
    cloudcover = st.slider("Cloud Cover (0 to 1)", 0.0, 1.0, value=0.5)
    visibility = st.number_input("Visibility (km)", value=10.0)
    solarradiation = st.number_input("Solar Radiation (W/m²)", value=500.0)
    uvindex = st.slider("UV Index", 0, 11, value=5)
    winddir_degree = st.slider("Wind Direction (Degrees)", 0, 360, value=90)

    winddir_sin = np.sin(np.deg2rad(winddir_degree))
    winddir_cos = np.cos(np.deg2rad(winddir_degree))

    # === Process Datetime Features ===
    dt = pd.to_datetime(datetime_input)
    datetime_features = {
        "hour": dt.hour,
        "dayofweek": dt.dayofweek,
        "quarter": dt.quarter,
        "month": dt.month,
        "year": dt.year,
        "dayofyear": dt.dayofyear,
        "dayofmonth": dt.day,
        "weekofyear": dt.isocalendar().week
    }

    # === Season One-Hots ===
    season_mapping = {
        'Spring': (3, 4, 5),
        'Summer': (6, 7, 8),
        'Winter': (12, 1, 2)
    }

    season_features = {
        "season_Spring": 0,
        "season_Summer": 0,
        "season_Winter": 0
    }

    for season, months in season_mapping.items():
        if dt.month in months:
            season_features[f"season_{season}"] = 1

    # === Month Name One-Hots ===
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_features = {f"month_name_{m}": 0 for m in month_names}
    month_features[f"month_name_{month_names[dt.month - 1]}"] = 1

    # === Weekend Feature ===
    week_type_feature = {
        "week_type_Weekend": 1 if dt.dayofweek >= 5 else 0
    }

    # === Lag and Rolling Features (NaNs for manual input) ===
    lag_features = {
        "rolling_mean_24hr": np.nan,
        "lag_1": np.nan,
        "lag_24": np.nan,
        "lag_168": np.nan,
        "rolling_std_24hr": np.nan,
        "rolling_mean": np.nan
    }

    # === Combine All Features ===
    final_input = {
        **{"datetime": datetime_input},
        **{
            "temp": temp,
            "dew": dew,
            "humidity": humidity,
            "windgust": windgust,
            "windspeed": windspeed,
            "sealevelpressure": sealevelpressure,
            "cloudcover": cloudcover,
            "visibility": visibility,
            "solarradiation": solarradiation,
            "uvindex": uvindex,
            "winddir_sin": winddir_sin,
            "winddir_cos": winddir_cos,
        },
        **datetime_features,
        **lag_features,
        **season_features,
        **month_features,
        **week_type_feature
    }

    final_columns = [
        'datetime', 'temp', 'dew', 'humidity', 'windgust', 'windspeed', 'sealevelpressure',
        'cloudcover', 'visibility', 'solarradiation', 'uvindex', 'winddir_sin', 'winddir_cos',
        'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
        'rolling_mean_24hr', 'lag_1', 'lag_24', 'lag_168', 'rolling_std_24hr', 'rolling_mean',
        'season_Spring', 'season_Summer', 'season_Winter',
        'month_name_Aug', 'month_name_Dec', 'month_name_Feb', 'month_name_Jan', 'month_name_Jul',
        'month_name_Jun', 'month_name_Mar', 'month_name_May', 'month_name_Nov', 'month_name_Oct',
        'month_name_Sep', 'week_type_Weekend'
    ]

    final_df = pd.DataFrame([final_input])[final_columns]
    X_pred = final_df.drop(columns=['datetime'])

    st.write("📋 Your Input Data")
    st.dataframe(X_pred)

    # 🔮 Prediction
    if st.button("🔮 Predict Energy Consumption"):
        # y_pred = model.predict(X_pred)
        # estimated_emission = y_pred[0] * EMISSION_FACTOR

        # st.success(f"**Predicted Energy Consumption:** {y_pred[0]:.2f} kWh")
        # st.info(f"**Estimated CO₂ Emissions:** {estimated_emission:.2f} kg")
        
        y_pred = model.predict(X_pred)
        estimated_emission = y_pred[0] * EMISSION_FACTOR  # Optional

        st.success(f"**Predicted Energy Consumption:** {y_pred[0]:.2f} kWh")
        st.info(f"**Estimated CO₂ Emissions:** {estimated_emission:.2f} kg")

        result_df = final_df.copy()
        result_df["Predicted Energy Consumption (kWh)"] = y_pred
        result_df["Estimated CO₂ Emissions (kg)"] = estimated_emission
        csv_result = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Prediction", csv_result, "datetime_energy_prediction.csv", "text/csv")        # 📥 Download Prediction
        
        # 🌱 Suggestions
        st.markdown("### 🌱 Suggestions to Reduce Carbon Emissions")
        st.markdown(
            """
            - **Use natural ventilation when possible.**
            - **Schedule regular maintenance of HVAC systems.**
            - **Monitor and optimize appliance usage.**
            """
        )


# Carbon Footprint Calculator
st.markdown("### 🌍 Carbon Footprint Calculator")
kWh_used = st.number_input("Enter the energy consumption (in kWh):", min_value=0.0, step=0.1)
if kWh_used > 0:
    carbon_emission = kWh_used * EMISSION_FACTOR
    st.write(f"💡 **Estimated CO₂ Emissions:** {carbon_emission:.2f} kg")
else:
    st.write("Please enter a valid energy consumption to calculate CO₂ emissions.")
