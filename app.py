import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = "final_model.h5"

st.set_page_config(
    page_title="Person Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS để làm giao diện đẹp hơn
st.markdown("""
    <style>
    /* Ẩn header và footer mặc định */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Container chính */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }

    /* Header styling */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Upload area styling */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }

    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Result container */
    .result-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }

    /* Warning message */
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }

    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def show_result(prob):
    is_person = prob >= 0.5
    confidence = prob if is_person else 1.0 - prob

    # Hiển thị kết quả với styling đẹp
    if is_person:
        st.markdown(f"""
            <div class="success-box">
                👤 PERSON DETECTED
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="warning-box">
                🚫 NON-PERSON
            </div>
        """, unsafe_allow_html=True)

    # Hiển thị độ tin cậy với progress bar
    st.markdown(f"<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown(f"**Độ tin cậy: {confidence * 100:.1f}%**")
    st.progress(confidence)
    st.markdown(f"</div>", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-title">
        🔍 Person Detector AI
    </div>
    <div class="header-subtitle">
        <strong>TRẦN HẢI NAM - 223332840</strong>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

# Main content area
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Ảnh")
    uploaded_file = st.file_uploader(
        "Kéo thả ảnh vào đây hoặc click để chọn",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Chọn ảnh định dạng JPG, JPEG hoặc PNG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Dự đoán ngay", type="primary", use_container_width=True):
            with st.spinner("🤖 AI đang phân tích ảnh..."):
                img_array = preprocess_image(image)
                prob = float(model.predict(img_array, verbose=0)[0][0])
                st.session_state.upload_result = prob
                st.rerun()
    else:
        st.info("👆 Vui lòng chọn ảnh để bắt đầu dự đoán")

with col2:
    st.markdown("### 📊 Kết quả dự đoán")
    st.markdown("<div class='result-container'>", unsafe_allow_html=True)

    if st.session_state.upload_result is not None:
        show_result(st.session_state.upload_result)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #6c757d;">
                <p style="font-size: 3rem;">📷</p>
                <p>Chưa có kết quả</p>
                <p style="font-size: 0.9rem;">Upload ảnh và nhấn "Dự đoán" để xem kết quả</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
