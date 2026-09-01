"""Streamlit demo for Skin Disease Classifier on Hugging Face Spaces."""

import streamlit as st
from PIL import Image
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from skin_disease.inference import SkinDiseasePredictor
from skin_disease.labels import get_severity

# Page config
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .severity-mild {
        background-color: #d4edda;
        color: #155724;
    }
    .severity-moderate {
        background-color: #fff3cd;
        color: #856404;
    }
    .severity-severe {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Skin Disease Classifier")

# Medical disclaimer
with st.container():
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Medical Disclaimer:</strong> This tool provides an AI-generated classification based on an uploaded image. 
    It is intended for <strong>research and decision-support purposes only</strong> and is <strong>NOT</strong> a substitute 
    for evaluation by a qualified healthcare professional. If you have a concerning or changing skin lesion or symptom, 
    seek professional medical advice.
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.write("""
    This classifier uses a hybrid deep learning model combining:
    - **ResNet50** (2048-d features)
    - **DenseNet121** (1024-d features)
    
    Trained to detect 22 skin disease categories.
    """)
    
    st.header("🎯 Severity Levels")
    st.write("""
    - 🟢 **MILD**: Common, non-serious conditions
    - 🟡 **MODERATE**: Requires attention
    - 🔴 **SEVERE**: Urgent medical evaluation recommended
    """)

# Load model with caching
@st.cache_resource
def load_predictor():
    """Load model from Hugging Face Hub."""
    try:
        return SkinDiseasePredictor(
            hf_repo_id="adarshRaj7/skin-disease-classifier",
            hf_filename="best_model.pt"
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin image (JPG, PNG, etc.)",
        type=["jpg", "jpeg", "png", "gif", "webp"]
    )

with col2:
    st.subheader("📊 Predictions")
    prediction_placeholder = st.empty()

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    with st.spinner("🔍 Analyzing image..."):
        predictor = load_predictor()
        if predictor:
            try:
                result = predictor.predict(image)
                
                # Display results
                st.success("✅ Analysis complete!")
                
                # Top prediction
                st.subheader("🎯 Top Prediction")
                top_class = result["predicted_class"]
                confidence = result["confidence"]
                severity = get_severity(top_class)
                
                # Color code by severity
                severity_color = {
                    "MILD": "🟢",
                    "MODERATE": "🟡",
                    "SEVERE": "🔴"
                }.get(severity, "⚪")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Disease", top_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    st.metric("Severity", f"{severity_color} {severity}")
                
                # All predictions (top 5)
                st.subheader("📈 Top 5 Predictions")
                predictions = result["all_predictions"][:5]
                
                for i, pred in enumerate(predictions, 1):
                    disease = pred["class"]
                    conf = pred["confidence"]
                    sev = get_severity(disease)
                    sev_icon = {
                        "MILD": "🟢",
                        "MODERATE": "🟡",
                        "SEVERE": "🔴"
                    }.get(sev, "⚪")
                    
                    # Progress bar
                    st.write(f"{i}. **{disease}** {sev_icon} {sev}")
                    st.progress(conf)
                    st.write(f"   Confidence: {conf:.1%}")
                
                # Grad-CAM visualization (if available)
                if "gradcam" in result and result["gradcam"] is not None:
                    st.subheader("🔥 Attention Map (Grad-CAM)")
                    st.write("Shows which parts of the image contributed most to the prediction:")
                    gradcam_img = Image.fromarray(result["gradcam"])
                    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        else:
            st.error("Could not load model")

else:
    st.info("👆 Upload an image to start analyzing")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>🔗 Model: <a href="https://huggingface.co/adarshRaj7/skin-disease-classifier">adarshRaj7/skin-disease-classifier</a></p>
    <p>💻 Source: <a href="https://github.com/adarshRaj7/hybrid-skin-disease-classifier">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
