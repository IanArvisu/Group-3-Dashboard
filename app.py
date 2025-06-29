import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_DIR = "models"
MODELS = {
    "Water Parameters Only": "cnn_lstm_water.h5",
    "Water + Climate Activity Parameters": "cnn_lstm_water+climate.h5",
    "Water + Volcanic Activity Parameters": "cnn_lstm_water+volcanic.h5",
    "Water + Climate + Volcanic Activity Parameters": "cnn_lstm_water+climate+volcanic.h5"
}
FEATURES = {
    "Water Parameters Only": ['Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)', 'pH Level', 'Dissolved Oxygen (mg/L)'],
    "Water + Climate Activity Parameters": ['Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)', 'pH Level', 'Dissolved Oxygen (mg/L)', 'RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION'],
    "Water + Volcanic Activity Parameters": ['Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)', 'pH Level', 'Dissolved Oxygen (mg/L)', 'CO2_Flux', 'SO2_Flux'],
    "Water + Climate + Volcanic Activity Parameters": [
        col for col in [
            'Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)',
            'pH Level', 'Dissolved Oxygen (mg/L)', 'RAINFALL', 'TMAX', 'TMIN', 'RH',
            'WIND_SPEED', 'WIND_DIRECTION', 'CO2_Flux', 'SO2_Flux'
        ] if col in pd.read_csv("cleaned_normalized_combined_data.csv").columns
    ]
}
TARGETS = ['Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 'Phosphate (mg/L)']
LOOKBACK = 12

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_normalized_combined_data.csv")

df = load_data()
df["Standardized_Date"] = pd.to_datetime(df["Standardized_Date"])
df = df.sort_values(["Location", "Standardized_Date"])
locations = df["Location"].unique().tolist()

# --- STREAMLIT UI ---
st.title("Water Quality Prediction Dashboard")
st.sidebar.header("Input Configuration")

input_type = st.sidebar.selectbox("Select input variable set:", list(MODELS.keys()))
selected_location = st.sidebar.selectbox("Select location:", locations)
interval = st.sidebar.radio("Select interval:", ["Monthly", "Yearly"])

filtered = df[df["Location"] == selected_location].copy()
features = FEATURES[input_type]

data_seq = filtered[features].tail(LOOKBACK)
if len(data_seq) < LOOKBACK:
    st.warning(f"Not enough data to predict. Need at least {LOOKBACK} time steps.")
else:
    X_input = np.expand_dims(data_seq.values, axis=0)
    model_path = os.path.join(MODEL_DIR, MODELS[input_type])

    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
    else:
        model = tf.keras.models.load_model(model_path, compile=False)
        prediction = model.predict(X_input)[0]

        st.subheader("Predicted Water Quality Levels")
        for i, target in enumerate(TARGETS):
            st.metric(label=target, value=f"{prediction[i]:.4f}")

        # --- Graph Interpretation ---
        st.subheader("📈 Recent Trends of Water Quality Parameters")
        fig, ax = plt.subplots(figsize=(8, 4))
        recent_dates = filtered["Standardized_Date"].tail(LOOKBACK)
        for feature in features:
            if feature in filtered.columns:
                ax.plot(recent_dates, filtered[feature].tail(LOOKBACK), label=feature)

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(f"Recent Parameter Trends at {selected_location}")
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True)
        st.pyplot(fig)

        # --- WQI Interpretation ---
        st.subheader("💧 Water Quality Index (WQI)")

        wqi_inputs = {
            "Dissolved Oxygen (mg/L)": filtered['Dissolved Oxygen (mg/L)'].iloc[-1] if 'Dissolved Oxygen (mg/L)' in filtered.columns else 7.0,
            "pH Level": filtered['pH Level'].iloc[-1] if 'pH Level' in filtered.columns else 7.0,
            "Ammonia (mg/L)": prediction[0],
            "Nitrate-N/Nitrite-N  (mg/L)": prediction[1],
            "Phosphate (mg/L)": prediction[2],
        }

        weights = {
            "Dissolved Oxygen (mg/L)": 0.3,
            "pH Level": 0.2,
            "Ammonia (mg/L)": 0.2,
            "Nitrate-N/Nitrite-N  (mg/L)": 0.15,
            "Phosphate (mg/L)": 0.15
        }

        subindex = []
        for param, value in wqi_inputs.items():
            if "Oxygen" in param:
                si = (value / 14) * 100
            elif "pH" in param:
                si = (1 - abs(7 - value) / 7) * 100
            else:
                si = max(0, 100 - (value * 10))
            subindex.append(si * weights[param])

        wqi = sum(subindex)

        if wqi < 50:
            status = "Excellent"
            interpretation = "Water is considered safe and clean, suitable for drinking and aquatic life."
            color = "🟢"
        elif wqi < 75:
            status = "Good"
            interpretation = "Water quality is slightly impacted but generally safe for most uses."
            color = "🟡"
        elif wqi < 90:
            status = "Fair"
            interpretation = "Water quality is moderately impacted. Not suitable for drinking without treatment."
            color = "🟠"
        else:
            status = "Poor"
            interpretation = "Water quality is severely impacted and generally unsafe."
            color = "🔴"

        st.metric(label="WQI Score", value=f"{wqi:.2f}", delta=status)
        st.info(f"{color} **Interpretation:** {interpretation}")
