import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Crop Disease Detector",
    layout="wide"
)
# Sidebar with logo and title
logo = Image.open("logo.png")
st.sidebar.image(logo, width=380)
st.sidebar.title("Crop Health AI")
st.sidebar.markdown("Local AI-based plant disease detection system.")
import streamlit as st





@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="theconqueror004/plant_disease_detection_model",
        filename="plant_disease_effnetb0.keras",
        token=st.secrets["HF_TOKEN"]
    )
    return tf.keras.models.load_model(model_path)

model = load_model()


with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

with open("disease_info.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

def calculate_visibility(img_array):
    gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

    gy, gx = np.gradient(gray)
    sharpness = np.var(gx) + np.var(gy)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    sharp_norm = min(sharpness / 50, 100)
    bright_norm = min(brightness / 2.55, 100)
    contrast_norm = min(contrast / 2.55, 100)

    visibility = (sharp_norm + bright_norm + contrast_norm) / 3
    return round(float(visibility), 2)

def predict_disease(img_array):
    img = Image.fromarray(img_array).resize((224, 224))
    img = np.array(img, dtype=np.float32)

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    class_name = idx_to_class[idx]
    confidence = float(preds[idx] * 100)

    return class_name, round(confidence, 2)

def compute_reliability(confidence, visibility):
    if confidence > 85 and visibility > 70:
        return "High"
    elif confidence > 70 and visibility > 60:
        return "Medium"
    else:
        return "Low"

def generate_report(img_array):
    visibility = calculate_visibility(img_array)
    pred_class, confidence = predict_disease(img_array)
    reliability = compute_reliability(confidence, visibility)

    info = disease_info.get(
        pred_class,
        {"symptoms": ["N/A"], "cause": "N/A", "risk": "N/A"}
    )

    report = f"""
### Disease Prediction Report

**Predicted Disease:** {pred_class}  
**Model Confidence:** {confidence:.2f}%  
**Image Visibility:** {visibility}%  
**Prediction Reliability:** {reliability}

**Observed Symptoms:**  
- {'; '.join(info['symptoms'])}

**Possible Cause:**  
- {info['cause']}

**Risk Conditions:**  
- {info['risk']}

---

This prediction is AI-assisted and intended for decision support only.
"""
    return report, confidence, visibility, reliability

st.sidebar.title("Crop Health AI")
st.sidebar.markdown("Local AI-based plant disease detection system.")

st.title("AI Crop Disease Detector")
st.write("Upload a leaf image to get disease prediction and insights.")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img_array, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            report, confidence, visibility, reliability = generate_report(img_array)

        st.subheader("Prediction Confidence")
        st.progress(int(min(confidence, 100)))

        st.markdown(f"**Reliability Level:** {reliability}")
        st.markdown(report)

        st.download_button(
            label="Download Report",
            data=report,
            file_name="crop_disease_report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("Developed locally using Streamlit and TensorFlow")
