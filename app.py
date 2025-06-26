import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="🌿 Rice Disease Detector",
    page_icon="🍃",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('leaf_disease_model.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

model = load_model()

CLASS_NAMES = [
    "Healthy", "Rust_Mild", "Rust_Severe",
    "Scab_Mild", "Scab_Severe", "Blight_Mild",
    "Blight_Severe", "Mildew_Mild", "Mildew_Severe"
]

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return (img_array / 255.0).astype(np.float32)[np.newaxis, ...]

def predict(image):
    processed_img = preprocess_image(image)
    pred = model.predict(processed_img)
    
    if isinstance(pred, list):  
        disease_idx = np.argmax(pred[0][0])
        severity = float(pred[1][0][0]) * 100  
    else:  
        disease_idx = np.argmax(pred[0])
        severity = 50.0  
    
    return CLASS_NAMES[disease_idx], severity

st.title("🍃 Rice Leaf Disease Classifier")
st.markdown("Upload an image of a rice leaf to check for diseases and severity")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file and model:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze"):
            with st.spinner("Detecting disease..."):
                disease, severity = predict(image)
                
                st.subheader("Results")
                st.metric("Disease", disease)
                st.metric("Severity", f"{severity:.1f}%")
                st.progress(int(severity))
                
                if "Healthy" in disease:
                    st.balloons()
                    st.success("This leaf appears healthy!")
                elif "Severe" in disease:
                    st.warning("⚠️ Immediate treatment needed!")
                    st.markdown("""
                    **Recommended Actions:**
                    - Apply fungicide (e.g., Azoxystrobin)
                    - Remove affected leaves
                    - Improve field drainage
                    """)

    with col2:
        st.markdown("""
        ### 📋 Disease Guide
        | Disease      | Symptoms |
        |--------------|----------|
        | Rust         | Orange pustules |
        | Scab         | Dark, scaly lesions |
        | Blight       | Water-soaked spots |
        | Powdery Mildew | White fungal growth |
        """)

if st.checkbox("Show debug info", False):
    st.write("Model input shape:", model.input_shape)
    st.write("Model output:", model.output_shape)
