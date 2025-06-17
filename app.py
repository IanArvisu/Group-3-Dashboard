import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Page Configuration ===
st.set_page_config(
    page_title="Water Quality Prediction Using Hybrid CNN-LSTM",
    layout="wide"
)

st.title("🌊 Water Quality Prediction Using Hybrid CNN-LSTM")
st.markdown("This app uses the latest 24 time steps of 10 environmental features to predict 5 water quality indicators.")

# === Sidebar Navigation ===
page = st.sidebar.radio("Navigation", [
    "Predicted Parameters",
    "Actual vs Predicted Graphs"
])

# === Load Model and Data ===
@st.cache_resource
def load_model_hybrid():
    return load_model("hybrid_model.h5")

@st.cache_data(ttl=0)  # Always load latest version
def load_data():
    df = pd.read_csv("preprocessed_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.drop(columns=["Date", "Location"], inplace=True, errors='ignore')
    return df

model = load_model_hybrid()
df = load_data()

# Debug: Print column list
st.write("✅ Columns in DataFrame:", df.columns.tolist())

# === Parameters to Predict ===
target_columns = [
    "Dissolved Oxygen (mg/L)",
    "pH Level",
    "Ammonia (mg/L)",
    "Nitrate-N/Nitrite-N  (mg/L)",
    "Phosphate (mg/L)"
]

# === Full feature list used for prediction ===
selected_features = [
    "Surface Water Temp (°C)",
    "Middle Water Temp (°C)",
    "Bottom Water Temp (°C)",
    "pH Level",
    "Ammonia (mg/L)",
    "Nitrate-N/Nitrite-N  (mg/L)",
    "Phosphate (mg/L)",
    "Dissolved Oxygen (mg/L)",
    "Month_sin",
    "Month_cos"
]

# === Function to Create Sequences ===
def create_sequences_multitarget(data, targets, time_steps=24):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i + time_steps])
        ys.append(targets[i + time_steps])
    return np.array(Xs), np.array(ys)

# === Page 1: Predicted Parameters ===
if page == "Predicted Parameters":
    st.subheader("🧪 Predicted Water Quality Parameters:")
    lookback = 24

    try:
        features = df[selected_features]
        last_24 = features.tail(lookback).copy()

        # Debug: Show input shape
        st.write("✅ Shape of selected features (last 24):", last_24.shape)

        if len(last_24) == lookback:
            X_input = last_24.values.reshape(1, lookback, -1)
            predicted = model.predict(X_input)[0]
            for i, param in enumerate(target_columns):
                st.metric(label=param, value=f"{predicted[i]:.3f}")
        else:
            st.warning(f"Not enough data to make prediction. Found only {len(last_24)} time steps.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# === Page 2: Actual vs Predicted Graphs ===
elif page == "Actual vs Predicted Graphs":
    st.subheader("📈 Actual vs Predicted: Water Quality Parameters")

    try:
        # Use full 10-feature input
        X_data = df[selected_features].values
        y_data = df[target_columns].values

        X_test, y_test = create_sequences_multitarget(X_data, y_data)
        y_pred = model.predict(X_test)

        for i, col in enumerate(target_columns):
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(x=range(len(y_test)), y=y_test[:, i], label="Actual", ax=ax)
            sns.lineplot(x=range(len(y_pred)), y=y_pred[:, i], label="Predicted", ax=ax, linestyle="--")
            ax.set_title(f"Actual vs. Predicted: {col}")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel(col)
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error in graph generation: {e}")
