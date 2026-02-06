import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = "final_model.h5"

st.set_page_config(
    page_title="Person Detector",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS để cải thiện giao diện
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .author-info {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin-top: 1rem;
    }
    .stButton>button {
        border-radius: 20px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
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
    confidence_percent = confidence * 100

    # Hiển thị kết quả với icon và màu sắc
    if is_person:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>👤 PHÁT HIỆN NGƯỜI</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>🚫 KHÔNG PHẢI NGƯỜI</h2>
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị độ tin cậy với progress bar
    st.markdown(f"""
    <div style='margin-top: 1rem;'>
        <p style='font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem; text-align: center;'>
            Độ tin cậy: {confidence_percent:.1f}%
        </p>
        <div style='background: #e0e0e0; border-radius: 15px; height: 30px; overflow: hidden;'>
            <div style='background: {'linear-gradient(90deg, #667eea 0%, #764ba2 100%)' if is_person else 'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)'};
                        width: {confidence_percent}%; height: 100%; display: flex; align-items: center; justify-content: center;
                        color: white; font-weight: bold; transition: width 0.5s;'>
                {confidence_percent:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị xác suất chi tiết
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Xác suất là người", f"{prob * 100:.1f}%")
    with col2:
        st.metric("Xác suất không phải người", f"{(1-prob) * 100:.1f}%")

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Thông tin")
    st.markdown("""
    **Ứng dụng:** Person Detector
    **Mô hình:** EfficientNet
    **Kích thước ảnh:** 224x224
    """)
    st.markdown("---")
    st.markdown("### 👨‍💻 Tác giả")
    st.markdown("**TRẦN HẢI NAM**")
    st.markdown("**MSSV:** 223332840")
    st.markdown("---")
    st.markdown("### 📝 Hướng dẫn")
    st.markdown("""
    1. Chọn ảnh từ máy tính
    2. Xem ảnh preview
    3. Nhấn nút **Dự đoán**
    4. Xem kết quả và độ tin cậy
    """)

# Header
st.markdown('<h1 class="main-header">👤 Person Detector</h1>', unsafe_allow_html=True)
st.markdown('<div class="author-info">TRẦN HẢI NAM - 223332840</div>', unsafe_allow_html=True)
st.markdown("---")

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

# Upload section
st.markdown("### 📸 Tải ảnh lên")
uploaded_file = st.file_uploader(
    "Chọn ảnh từ máy tính của bạn...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Hỗ trợ định dạng: JPG, JPEG, PNG"
)

# Main content columns
col_img, col_result = st.columns([3, 2])

with col_img:
    st.markdown("### 🖼️ Ảnh đã tải lên")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Ảnh của bạn")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
                with st.spinner("⏳ Đang xử lý và dự đoán..."):
                    img_array = preprocess_image(image)
                    prob = float(model.predict(img_array, verbose=0)[0][0])
                    st.session_state.upload_result = prob
                    st.rerun()
    else:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu")

with col_result:
    st.markdown("### 📊 Kết quả dự đoán")
    with st.container(border=True):
        if st.session_state.upload_result is not None:
            show_result(st.session_state.upload_result)
        else:
            st.info("💡 Kết quả sẽ hiển thị ở đây sau khi bạn nhấn nút 'Dự đoán'")
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #999;'>
                <p style='font-size: 3rem; margin: 0;'>📈</p>
                <p>Chờ dự đoán...</p>
            </div>
            """, unsafe_allow_html=True)
