import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================
SEQUENCE_LENGTH = 2000
THRESHOLD = 0.5

st.set_page_config(
    page_title="EXO-SCAN AI",
    page_icon="🪐",
    layout="wide"
)

# ======================================================
# CUSTOM ATTENTION LAYER (REQUIRED)
# ======================================================
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")
        self.u = self.add_weight(shape=(self.units,),
                                 initializer="glorot_uniform")

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, -1)
        return tf.reduce_sum(x * ait, axis=1)

# ======================================================
# LOAD MODEL
# ======================================================

import os

MODEL_PATH = "F:\\newworldbegins\\exonew\\Streamlit\\exoplanet_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer}
    )

model = load_model()


# ======================================================
# STYLES (SCI-FI)
# ======================================================
st.markdown("""
<style>
body { background-color: #050816; color: #e0e6ff; }
h1, h2, h3 { color: #7df9ff; }
.stButton>button {
    background: linear-gradient(90deg, #7f00ff, #00c6ff);
    color: white;
    border-radius: 12px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HELPERS
# ======================================================
def fix_length(arr, target):
    if len(arr) > target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)))

# ======================================================
# HEADER
# ======================================================
st.title("🛸 EXO-SCAN AI")
st.caption("Hybrid CNN-BiLSTM with Attention • NASA-grade Exoplanet Detection")

st.divider()

# ======================================================
# UPLOAD
# ======================================================
files = st.file_uploader(
    "📡 Upload Light-Curve CSV Files (time, flux)",
    type=["csv"],
    accept_multiple_files=True
)

if files:
    for file in files:
        st.subheader(f"🪐 Target: `{file.name}`")

        df = pd.read_csv(file)

        if "flux" not in df.columns:
            st.error("❌ CSV must contain a `flux` column")
            continue

        flux = fix_length(df["flux"].values, SEQUENCE_LENGTH)
        X = flux.reshape(1, SEQUENCE_LENGTH, 1)

        # Prediction
        prob = float(model.predict(X, verbose=0)[0][0])
        confidence = prob * 100

        col1, col2 = st.columns([2, 1])

        # Plot
        with col1:
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(flux, color="#00e5ff", linewidth=1.2)
            ax.set_facecolor("#050816")
            ax.set_title("Light Curve Signal", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig)

        # Result Panel
        with col2:
            st.markdown("### 🧠 AI VERDICT")
            if prob >= THRESHOLD:
                st.success("🌍 EXOPLANET DETECTED")
            else:
                st.error("☄️ NO TRANSIT SIGNAL")

            st.metric("Detection Confidence", f"{confidence:.2f}%")
            st.progress(prob)

        st.divider()
else:
    st.info("👆 Upload Kepler/TESS light-curve CSVs to begin scan")

# ======================================================
# FOOTER
# ======================================================
st.caption("EXO-SCAN AI • Deep Learning for Space Discovery")
