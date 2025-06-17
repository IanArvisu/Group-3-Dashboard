import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Streamlit Page Config ===
st.set_page_config(
    page_title="Water Quality Prediction Dashboard",
    layout="wide"
)

st.title("🌊 Water Quality Prediction Using CNN, LSTM, and Hybrid Models")
st.markdown("This dashboard allows prediction of water quality parameters using 24-step environmental sequences and displays results from multiple models.")

# === Model Loader ===
@st.cache_resource
def load_model_file(name):
    return load_model(name)

# === Preprocess Data Exactly Like Training ===
@st.cache_data(ttl=0)
def load_and_preprocess_data():
    df = pd.read_csv("preprocessed_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df.drop(columns=["Date", "Location", "Month"], inplace=True, errors="ignore")

    target_columns = [
        "Dissolved Oxygen (mg/L)",
        "pH Level",
        "Ammonia (mg/L)",
        "Nitrate-N/Nitrite-N  (mg/L)",
        "Phosphate (mg/L)"
    ]

    for col in df.columns:
        if col not in target_columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)
            df[f"{col}_roll3"] = df[col].rolling(3).mean()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    selected_features = [col for col in df.columns if col not in target_columns]
    return df, selected_features, target_columns

df, selected_features, target_columns = load_and_preprocess_data()

# === Create Sequences ===
def create_sequences_multitarget(data, targets, time_steps=24):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i+time_steps])
        ys.append(targets[i+time_steps])
    return np.array(Xs), np.array(ys)

# === Sidebar for Navigation ===
page = st.sidebar.radio("Navigation", ["Predicted Parameters", "Actual vs Predicted Graphs"])
model_name = st.sidebar.selectbox("Select Model", ["CNN", "LSTM", "Hybrid"])
model_file = {
    "CNN": "cnn_model.h5",
    "LSTM": "lstm_model.h5",
    "Hybrid": "hybrid_model.h5"
}[model_name]

# === Load Selected Model ===
try:
    model = load_model_file(model_file)
except Exception as e:
    st.error(f"❌ Could not load model '{model_file}': {e}")
    st.stop()

# === Page 1: Live Prediction ===
if page == "Predicted Parameters":
    st.subheader("🧪 Live Predicted Water Quality Parameters:")
    lookback = 24

    try:
        last_24 = df[selected_features].tail(lookback).copy()
        if len(last_24) == lookback:
            X_input = last_24.values.reshape(1, lookback, len(selected_features))
            prediction = model.predict(X_input)[0]
            for i, target in enumerate(target_columns):
                st.metric(label=target, value=f"{prediction[i]:.3f}")
        else:
            st.warning(f"Not enough data to predict. Require {lookback} timesteps.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# === Page 2: Actual vs Predicted Graphs ===
elif page == "Actual vs Predicted Graphs":
    st.subheader(f"📈 Actual vs Predicted Graphs — Model: {model_name}")

    try:
        X_data = df[selected_features].values
        y_data = df[target_columns].values
        X_test, y_test = create_sequences_multitarget(X_data, y_data, time_steps=24)
        y_pred = model.predict(X_test)

        for i, target in enumerate(target_columns):
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(x=range(len(y_test)), y=y_test[:, i], label="Actual", ax=ax)
            sns.lineplot(x=range(len(y_pred)), y=y_pred[:, i], label="Predicted", ax=ax, linestyle="--")
            ax.set_title(f"{target}")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel(target)
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Graph generation failed: {e}")
