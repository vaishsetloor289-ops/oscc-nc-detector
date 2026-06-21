import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import os
import gdown

st.set_page_config(page_title="OSCC N:C Ratio Detector", layout="centered")

MODEL_PATH = "oscc_nc_model.h5"
THRESHOLD_PATH = "threshold.json"
MODEL_FILE_ID = "1eNc0jyUXZeW8L_zBVq53kXkX_-OLC_2B"

@st.cache_resource
def load_model_and_threshold():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only, ~220MB)..."):
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            threshold = json.load(f)["threshold"]
    else:
        threshold = 0.223  # fallback default if file missing

    return model, threshold

model, THRESHOLD = load_model_and_threshold()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

st.markdown(
    "<p style='text-align: center; font-size: 20px; color: gray;'>"
    "Government Dental College and Research Institute, Bengaluru<br>"
    "Department of Oral Pathology &amp; Microbiology</p>",
    unsafe_allow_html=True
)

st.title("AI-Based N:C Ratio Detection in Oral Squmaous Cell Carcinoma")
st.markdown("Upload a 40x H&E-stained oral histopathology image to assess nuclear-cytoplasmic ratio.")
st.info("This is a pilot research tool developed for academic and proof-of-concept purposes. It is not intended for clinical diagnosis.")

with st.expander("CLICK HERE : What is N:C Ratio, and why does it matter?"):
    st.markdown("""
The **Nuclear-Cytoplasmic (N:C) ratio** is the proportion of a cell's nucleus size relative to its cytoplasm size.

In **normal, healthy oral epithelial cells**, the nucleus is small and the cytoplasm is abundant — giving a **low N:C ratio**.

In **Oral Squamous Cell Carcinoma (OSCC)**, malignant cells often show:
- Enlarged, irregular nuclei
- Reduced cytoplasm
- Hyperchromatism (darker-staining nuclei)

This results in a **visibly increased N:C ratio**, which is one of the key cytological features pathologists assess when identifying dysplastic or malignant changes under the microscope.

**Why it matters:** N:C ratio assessment is traditionally subjective and can vary between observers. An AI-assisted tool aims to provide a more objective, reproducible first-pass assessment — supporting, not replacing, expert pathologist evaluation.
    """)

uploaded_file = st.file_uploader("Upload histopathology image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array_exp)[0][0]

    is_normal = prediction > THRESHOLD
    label = "Normal N:C Ratio" if is_normal else "Increased N:C Ratio (OSCC-consistent)"
    confidence = prediction if is_normal else (1 - prediction)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if is_normal:
            st.success(f"### Result: {label}")
        else:
            st.error(f"### Result: {label}")
    with col2:
        st.metric("Confidence", f"{confidence*100:.1f}%")

    st.markdown("### Model Interpretability (Grad-CAM)")
    st.caption("Highlighted regions show where the model focused its prediction.")

    last_conv_layer = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array_exp, model, last_conv_layer)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    original_img = (img_array * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    col3, col4 = st.columns(2)
    with col3:
        st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_container_width=True)
    with col4:
        st.image(overlay, caption="Overlay", use_container_width=True)

    st.markdown("---")
    st.caption("Developed as part of a pilot study: 'Evaluating an AI for Automated Nuclear-Cytoplasmic Ratio Detection in OSCC' — Government Dental College and Research Institute, Bengaluru.")
    st.caption("Authors: Swasti Haswani, Dr. Vaishnavi Setloor")
    st.caption("Under the guidance of Dr. Sahana Srinath")
