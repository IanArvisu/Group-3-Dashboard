import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Streamlit Settings ===
st.set_page_config("Water Quality Dashboard", layout="wide")

# === Load Model File ===
@st.cache_resource
def load_model_file(model_path):
    return load_model(model_path)

# === Preprocessing Function ===
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

    features = [col for col in df.columns if col not in target_columns]
    return df, features, target_columns

# === Create Sequences ===
def create_sequences(data, targets, time_steps=24):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i+time_steps])
        ys.append(targets[i+time_steps])
    return np.array(Xs), np.array(ys)

# === Load data ===
df, feature_columns, target_columns = load_and_preprocess_data()

# === Sidebar ===
model_name = st.sidebar.selectbox("Select Model", ["CNN", "LSTM", "Hybrid"])
page = st.sidebar.radio("Navigation", ["Live Parameters", "Actual vs Predicted", "Metrics Table"])

model_path = {
    "CNN": "cnn_model.h5",
    "LSTM": "lstm_model.h5",
    "Hybrid": "hybrid_model.h5"
}[model_name]

# === Load model ===
try:
    model = load_model_file(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# === Shared prediction block ===
lookback = 24
X_all = df[feature_columns].values
y_all = df[target_columns].values
X_seq, y_seq = create_sequences(X_all, y_all, time_steps=lookback)

try:
    y_pred = model.predict(X_seq)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

# === Page 1: Live Parameters ===
if page == "Live Parameters":
    st.header("🧪 Live Predicted Water Quality Parameters")
    if df.shape[0] >= lookback:
        last_input = df[feature_columns].tail(lookback).values.reshape(1, lookback, len(feature_columns))
        try:
            live_pred = model.predict(last_input)[0]
            for i, col in enumerate(target_columns):
                st.metric(label=col, value=f"{live_pred[i]:.3f}")
        except Exception as e:
            st.error(f"❌ Live prediction failed: {e}")
    else:
        st.warning("Not enough data for prediction.")

# === Page 2: Actual vs Predicted ===
elif page == "Actual vs Predicted":
    st.header(f"📉 Actual vs Predicted Graphs — {model_name}")
    for i, target in enumerate(target_columns):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=range(len(y_seq)), y=y_seq[:, i], label="Actual", ax=ax)
        sns.lineplot(x=range(len(y_pred)), y=y_pred[:, i], label="Predicted", ax=ax, linestyle="--")
        ax.set_title(f"{target}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(target)
        ax.grid(True)
        st.pyplot(fig)

# === Page 3: Metrics Table ===
elif page == "Metrics Table":
    st.header(f"📊 Evaluation Metrics — {model_name}")
    metrics = {
        "Parameter": [],
        "MAE": [],
        "MSE": [],
        "R² Score": []
    }

    for i, col in enumerate(target_columns):
        y_true = y_seq[:, i]
        y_hat = y_pred[:, i]
        metrics["Parameter"].append(col)
        metrics["MAE"].append(round(mean_absolute_error(y_true, y_hat), 4))
        metrics["MSE"].append(round(mean_squared_error(y_true, y_hat), 4))
        metrics["R² Score"].append(round(r2_score(y_true, y_hat), 4))

    st.dataframe(pd.DataFrame(metrics), use_container_width=True)
