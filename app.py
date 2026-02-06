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
    initial_sidebar_state="collapsed",
)

# Custom CSS để cải thiện giao diện
st.markdown("""
    <style>
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Card styling */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }

    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
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
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }

    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
        padding: 1rem;
    }

    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Result badge */
    .result-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .person-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }

    .non-person-badge {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }

    /* Spacing improvements */
    .stMarkdown {
        margin-bottom: 1rem;
    }

    /* Container improvements */
    [data-testid="stVerticalBlock"] {
        gap: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

    # Hiển thị badge kết quả
    badge_class = "person-badge" if is_person else "non-person-badge"
    badge_text = "👤 PERSON" if is_person else "🚫 NON-PERSON"

    st.markdown(f"""
        <div class="result-badge {badge_class}">
            {badge_text}
        </div>
    """, unsafe_allow_html=True)

    # Hiển thị độ tin cậy với styling đẹp
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="🎯 Độ tin cậy",
        value=f"{confidence * 100:.1f}%",
        delta=f"{'Cao' if confidence >= 0.8 else 'Trung bình' if confidence >= 0.6 else 'Thấp'}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Progress bar để hiển thị trực quan
    st.progress(confidence, text=f"Xác suất: {confidence * 100:.1f}%")

# Header với gradient
st.markdown("""
    <div class="main-header">
        <h1>👤 Person Detector</h1>
        <p>TRẦN HẢI NAM - 223332840</p>
    </div>
""", unsafe_allow_html=True)

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

# Container chính
st.markdown("### 📸 Tải ảnh lên")
uploaded_file = st.file_uploader(
    "Chọn ảnh để phân tích (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Tải lên ảnh chứa người hoặc đối tượng khác để phân tích"
)

# Layout với 2 cột
col_img, col_result = st.columns([1.2, 1], gap="large")

with col_img:
    st.markdown("#### 🖼️ Ảnh đã tải lên")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
            with st.spinner("⏳ Đang phân tích ảnh..."):
                img_array = preprocess_image(image)
                prob = float(model.predict(img_array, verbose=0)[0][0])
                st.session_state.upload_result = prob
                st.rerun()
    else:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu phân tích")

with col_result:
    st.markdown("#### 📊 Kết quả dự đoán")
    with st.container(border=True):
        if st.session_state.upload_result is not None:
            show_result(st.session_state.upload_result)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <p style="font-size: 3rem; margin: 0;">📈</p>
                <p style="margin-top: 1rem;">Kết quả sẽ hiển thị ở đây sau khi bạn nhấn nút "Dự đoán"</p>
            </div>
            """, unsafe_allow_html=True)
