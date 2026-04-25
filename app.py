import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# MODERN UI CSS
# =====================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
    color: white;
}

/* Hide Streamlit default */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
    color: white;
}

.sub-title {
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:25px;
}

/* Glass cards */
.glass {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 25px;
    border-radius: 20px;
    text-align:center;
    color:white;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
}

.pred-label {
    font-size: 14px;
    letter-spacing: 1px;
    opacity:0.9;
}

.pred-class {
    font-size: 34px;
    font-weight: 800;
    margin-top:8px;
}

.conf {
    font-size:18px;
    margin-top:10px;
}

/* uploader */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 18px;
    border: 1px dashed rgba(255,255,255,0.25);
}

/* metric chips */
.chip {
    background: rgba(255,255,255,0.08);
    padding:10px 16px;
    border-radius:999px;
    display:inline-block;
    margin-right:8px;
    margin-bottom:8px;
    font-size:14px;
}

/* section title */
.sec {
    font-size:22px;
    font-weight:700;
    margin-bottom:10px;
    margin-top:10px;
}

/* image captions */
.cap {
    text-align:center;
    color:#cbd5e1;
    margin-top:6px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CONSTANTS
# =====================================================
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

SEG_IMG_SIZE = 256
RED_IMG_SIZE = 128

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    unet = tf.keras.models.load_model("unet_final.h5", compile=False)
    cnn = tf.keras.models.load_model("cnn_final.h5", compile=False)
    return unet, cnn

unet_model, cnn_model = load_models()

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="main-title">🧠 Brain Tumor Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">U-Net Segmentation + CNN Classification for MRI Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">
<span class="chip">⚡ Fast Prediction</span>
<span class="chip">🎯 Multi-Class Detection</span>
<span class="chip">🩺 MRI Analysis</span>
<span class="chip">🤖 AI Powered</span>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# UPLOADER
# =====================================================
uploaded_file = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# MAIN
# =====================================================
if uploaded_file is not None:

    pil_img = Image.open(uploaded_file)
    img_rgb = np.array(pil_img)

    if len(img_rgb.shape) == 2:
        gray = img_rgb
    else:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.uint8)

    # ------------------------------------------
    # U-NET INPUT
    # ------------------------------------------
    img_unet = cv2.resize(gray, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    img_unet = img_unet.astype(np.float32) / 255.0
    img_unet = np.expand_dims(img_unet, axis=-1)

    pred_mask = unet_model.predict(
        np.expand_dims(img_unet, axis=0),
        verbose=0
    )[0]

    mask = (pred_mask > 0.5).astype(np.uint8)

    # ------------------------------------------
    # OVERLAY
    # ------------------------------------------
    overlay = (img_unet.squeeze() * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    overlay[mask.squeeze() == 1] = [255, 0, 0]

    extracted = img_unet.squeeze() * mask.squeeze()

    # ------------------------------------------
    # CNN INPUT
    # ------------------------------------------
    cnn_input = cv2.resize(gray, (RED_IMG_SIZE, RED_IMG_SIZE))
    cnn_input = cnn_input.astype(np.float32) / 255.0
    cnn_input = cnn_input.reshape(1,128,128,1)

    pred = cnn_model.predict(cnn_input, verbose=0)[0]

    class_idx = np.argmax(pred)
    pred_class = classes[class_idx]
    confidence = float(pred[class_idx])

    # =================================================
    # RESULT CARD
    # =================================================
    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-label">PREDICTED CLASS</div>
        <div class="pred-class">{pred_class.upper()}</div>
        <div class="conf">Confidence: {confidence:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =================================================
    # IMAGES SECTION
    # =================================================
    st.markdown('<div class="sec">📊 MRI Visual Analysis</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(gray, use_container_width=True)
        st.markdown('<div class="cap">Original MRI</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.markdown('<div class="cap">Tumor Highlight Overlay</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(extracted, use_container_width=True)
        st.markdown('<div class="cap">Extracted Tumor Region</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # =================================================
    # CHART SECTION
    # =================================================
    st.markdown('<div class="sec">📈 Prediction Confidence</div>', unsafe_allow_html=True)

    chart_data = {
        classes[i]: float(pred[i])
        for i in range(len(classes))
    }

    st.bar_chart(chart_data)

    # =================================================
    # STATUS MESSAGE
    # =================================================
    if np.sum(mask) == 0:
        st.warning("⚠️ No tumor region strongly detected by segmentation model.")

# =====================================================
# FOOTER
# =====================================================
st.write("")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:14px; padding:20px;'>
Built for Final Year Project • Streamlit • Deep Learning • Medical Imaging
</div>
""", unsafe_allow_html=True)