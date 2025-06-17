import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config("Water Quality Dashboard", layout="wide")

@st.cache_resource
def load_model_file(model_path):
    return load_model(model_path)

@st.cache_data(ttl=0)
def load_and_preprocess_data():
    df = pd.read_csv("preprocessed_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    target_columns = [
        "Dissolved Oxygen (mg/L)",
        "pH Level",
        "Ammonia (mg/L)",
        "Nitrate-N/Nitrite-N  (mg/L)",
        "Phosphate (mg/L)"
    ]

    grouped = df.groupby("Location")
    all_processed = {}

    for loc, group in grouped:
        group = group.sort_values("Date").copy()
        group.drop(columns=["Month"], inplace=True, errors="ignore")
        for col in group.columns:
            if col not in target_columns + ["Date", "Location"]:
                group[f"{col}_lag1"] = group[col].shift(1)
                group[f"{col}_lag2"] = group[col].shift(2)
                group[f"{col}_roll3"] = group[col].rolling(3).mean()
        group.replace([np.inf, -np.inf], np.nan, inplace=True)
        group.dropna(inplace=True)
        group.reset_index(drop=True, inplace=True)
        all_processed[loc] = group

    return all_processed, target_columns

def create_sequences(data, targets, time_steps=24):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i+time_steps])
        ys.append(targets[i+time_steps])
    return np.array(Xs), np.array(ys)

def compute_wqi(subset):
    norm_cols = subset.copy()
    for col in norm_cols.columns:
        norm_cols[col] = (norm_cols[col] - norm_cols[col].min()) / (norm_cols[col].max() - norm_cols[col].min())
    return norm_cols.mean(axis=1)

# === Sidebar Controls ===
data_by_location, target_columns = load_and_preprocess_data()

model_name = st.sidebar.selectbox("Select Model", ["CNN", "LSTM", "Hybrid"])
location = st.sidebar.selectbox("Select Location", list(data_by_location.keys()))
interval = st.sidebar.selectbox("Aggregation Interval", ["Weekly", "Monthly", "Yearly"])
page = st.sidebar.radio("Navigation", ["Live Parameters", "Actual vs Predicted", "Metrics Table"])

model_path = {
    "CNN": "cnn_model.h5",
    "LSTM": "lstm_model.h5",
    "Hybrid": "hybrid_model.h5"
}[model_name]

# === Load Model ===
try:
    model = load_model_file(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# === Prepare Data ===
df = data_by_location[location]
lookback = 24

if df.shape[0] <= lookback:
    st.error("❌ Not enough data for prediction. Please select a different location.")
    st.stop()

feature_columns = [col for col in df.columns if col not in target_columns + ["Date", "Location"]]
X_all = df[feature_columns].values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
y_all = df[target_columns].values
X_seq, y_seq = create_sequences(X_all, y_all, time_steps=lookback)

# === Run Prediction ===
try:
    y_pred = model.predict(X_seq)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.write("X_seq shape:", X_seq.shape)
    st.write("Any NaNs:", np.isnan(X_seq).any())
    st.write("Preview of X_seq[0]:", X_seq[0] if len(X_seq) else "Empty")
    st.stop()

# === Postprocess ===
df_pred = df.iloc[lookback:].copy()
df_pred[target_columns] = y_seq
for i, col in enumerate(target_columns):
    df_pred[f"Predicted_{col}"] = y_pred[:, i]

df_pred["WQI"] = compute_wqi(df_pred[target_columns])
df_pred["Predicted_WQI"] = compute_wqi(df_pred[[f"Predicted_{col}" for col in target_columns]])

if interval == "Weekly":
    df_agg = df_pred.resample("W", on="Date").mean(numeric_only=True)
elif interval == "Monthly":
    df_agg = df_pred.resample("M", on="Date").mean(numeric_only=True)
else:
    df_agg = df_pred.resample("Y", on="Date").mean(numeric_only=True)

# === Page 1 ===
if page == "Live Parameters":
    st.header("🧪 Live Predicted Water Quality Parameters")
    last_input = df[feature_columns].tail(lookback).values.reshape(1, lookback, len(feature_columns))

    try:
        live_pred = model.predict(last_input)[0]
        wqi_inputs = {}
        for i, col in enumerate(target_columns):
            value = live_pred[i]
            st.metric(label=col, value=f"{value:.3f}")
            wqi_inputs[col] = value

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

        st.markdown("---")
        st.subheader("💧 Water Quality Index (WQI)")
        st.metric(label="WQI Score", value=f"{wqi:.2f}", delta=status)
        st.info(f"{color} **Interpretation:** {interpretation}")

    except Exception as e:
        st.error(f"❌ Live prediction failed: {e}")

elif page == "Actual vs Predicted":
    st.header(f"📈 Actual vs Predicted Graphs — {model_name} ({location})")
    st.caption(f"Aggregation Interval: **{interval}**")
    
    for col in target_columns + ["WQI"]:
        fig, ax = plt.subplots(figsize=(10, 4))
        actual = df_agg[col]
        predicted = df_agg.get(f"Predicted_{col}")
        
        if predicted is not None:
            sns.lineplot(data=actual, label="Actual", ax=ax)
            sns.lineplot(data=predicted, label="Predicted", ax=ax, linestyle="--")

        ax.set_title(f"{col} ({interval})")
        ax.set_xlabel(f"Date")
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


# === Page 3 ===
elif page == "Metrics Table":
    st.header(f"📊 Evaluation Metrics — {model_name} ({location})")
    metrics = {"Parameter": [], "MAE": [], "MSE": [], "R² Score": []}
    for i, col in enumerate(target_columns):
        y_true = y_seq[:, i]
        y_hat = y_pred[:, i]
        metrics["Parameter"].append(col)
        metrics["MAE"].append(round(mean_absolute_error(y_true, y_hat), 4))
        metrics["MSE"].append(round(mean_squared_error(y_true, y_hat), 4))
        metrics["R² Score"].append(round(r2_score(y_true, y_hat), 4))

    st.dataframe(pd.DataFrame(metrics), use_container_width=True)
