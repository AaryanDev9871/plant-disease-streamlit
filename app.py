import html
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
CLASS_PATH = BASE_DIR / "class_names.json"
MAX_SIZE = 10 * 1024 * 1024  # 10 MB

# Model Registry mapped to your exact filenames
MODEL_PATHS = {
    "MobileNetV2": BASE_DIR / "plant_disease_model.keras",
    "Custom CNN": BASE_DIR / "custom_cnn_best_validation_10epoch.keras",
    "EfficientNetB0": BASE_DIR / "efficientnetb0_best_validation_10epoch.keras",
}

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --page: #f5f7f1;
        --ink: #25382b;
        --muted: #5d6c60;
        --green: #356b43;
        --leaf: #4e8c5b;
        --border: #e1e8dd;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--page) !important;
        color: var(--ink);
        font-family: "DM Sans", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1440px;
        padding: 0 0 30px !important;
    }

    /* HTML header */
    .navbar {
        box-sizing: border-box;
        width: 100%;
        min-height: 82px;
        padding: 18px 8%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        border-bottom: 1px solid #e0e7dc;
        background: #ffffff;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        color: #24382a;
        font: 800 19px "Manrope", sans-serif;
        letter-spacing: -.5px;
    }

    .brand-icon {
        color: #4e8659;
        font-size: 27px;
    }

    .college-info {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 5px;
        text-align: right;
    }

    .college-info strong {
        color: #26392b;
        font-size: 13px;
    }

    .college-info span {
        color: #617264;
        font-size: 11px;
    }

    /* Hero */
    .hero {
        max-width: 900px;
        padding: 58px 24px 45px;
        margin: 0 auto;
        text-align: center;
        background: transparent;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 11px 15px;
        border: 1px solid #dce7d8;
        border-radius: 50px;
        background: #ffffff;
        color: #467a4f;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.7px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        display: inline-block;
        border-radius: 50%;
        background: #5d9b68;
    }

    .hero h1 {
        margin: 23px 0 17px;
        color: #25382b;
        font: 800 clamp(38px, 5.8vw, 61px)/1.13 "Manrope", sans-serif;
        letter-spacing: -2.5px;
    }

    .hero h1 span {
        color: #4e8c5b;
    }

    .subtitle {
        max-width: 570px;
        margin: 0 auto;
        color: #5d6c60;
        font-size: 15px;
        line-height: 1.8;
    }

    /* Streamlit layout */
    div[data-testid="stHorizontalBlock"] {
        gap: 24px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #e1e8dd !important;
        border-radius: 17px !important;
        box-shadow: 0 10px 32px #233a2710;
        padding: 22px;
    }

    .section-heading {
        min-height: 48px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }

    .step-label {
        color: #467a4f;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.7px;
    }

    .section-heading h2 {
        margin: 7px 0 0;
        color: #293b2e;
        font: 700 17px "Manrope", sans-serif;
    }

    .file-type {
        color: #707e72;
        font-size: 9px;
        white-space: nowrap;
    }

    .result-tag {
        padding: 8px 10px;
        border-radius: 7px;
        background: #edf4e9;
        color: #3d6d45;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .5px;
        white-space: nowrap;
    }

    /* Uploader */
    div[data-testid="stFileUploader"] {
        padding: 14px;
        border: 1.5px dashed #b9cdb6;
        border-radius: 12px;
        background: #f6f9f3 !important;
    }

    div[data-testid="stFileUploader"] section {
        border: none;
        background: transparent;
    }

    div[data-testid="stFileUploader"] button {
        border: 1px solid #cbdcc6;
        border-radius: 8px;
        background: #ffffff;
        color: #356b43;
    }

    div[data-testid="stImage"] img {
        max-height: 280px;
        object-fit: contain;
        border-radius: 9px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 47px;
        border: 0;
        border-radius: 9px;
        background: #356b43;
        color: #ffffff;
        font-family: "DM Sans", sans-serif;
        font-size: 13px;
        font-weight: 700;
    }

    .stButton > button:hover:not(:disabled) {
        background: #285735;
        color: #ffffff;
    }

    .stButton > button:disabled {
        background: #9bb39d;
        color: #ffffff;
    }

    .privacy-note {
        margin: 13px 0 0;
        color: #6e7b70;
        text-align: center;
        font-size: 10px;
    }

    /* Empty results */
    .result-empty {
        box-sizing: border-box;
        min-height: 335px;
        padding: 24px 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        background: #ffffff;
        color: #25382b;
    }

    .result-placeholder {
        width: 66px;
        height: 66px;
        display: grid;
        place-items: center;
        margin-bottom: 21px;
        border-radius: 19px;
        background: #edf3e8;
        color: #47774d;
        font: 700 23px "Manrope", sans-serif;
    }

    .result-empty strong {
        color: #293c2e;
        font-size: 13px;
    }

    .result-empty p {
        margin: 10px 0 22px;
        color: #69796b;
        font-size: 12px;
        line-height: 1.7;
    }

    .result-feature {
        box-sizing: border-box;
        width: 100%;
        max-width: 280px;
        padding: 12px;
        margin-top: 7px;
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid #e6ece2;
        border-radius: 8px;
        background: #ffffff;
        color: #4f6252;
        text-align: left;
        font-size: 10px;
    }

    .result-feature span {
        color: #477c50;
        font-weight: 700;
    }

    /* Prediction result */
    .result-content {
        min-height: 335px;
        padding: 18px 8px;
        background: #ffffff;
        color: #25382b;
    }

    .result-symbol {
        width: 48px;
        height: 48px;
        display: grid;
        place-items: center;
        border-radius: 14px;
        background: #e7f2e3;
        color: #347345;
        font-size: 23px;
    }

    .result-overline {
        display: block;
        margin-top: 24px;
        color: #526b56;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.4px;
    }

    .prediction-name {
        margin: 8px 0 12px;
        overflow-wrap: anywhere;
        color: #26392b;
        font: 700 23px/1.4 "Manrope", sans-serif;
    }

    .model-badge {
        display: inline-block;
        margin-bottom: 20px;
        padding: 4px 10px;
        background: #edf4e9;
        border-radius: 6px;
        color: #3d6d45;
        font-size: 11px;
        font-weight: 700;
    }

    .compare-card {
        padding: 14px;
        margin-bottom: 12px;
        border: 1px solid #e1e8dd;
        border-radius: 12px;
        background: #fafcf8;
    }

    .compare-title {
        font-size: 12px;
        font-weight: 700;
        color: #2d4533;
        margin-bottom: 4px;
    }

    .compare-pred {
        font-size: 15px;
        font-weight: 700;
        color: #26392b;
        margin-bottom: 8px;
    }

    .confidence-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        color: #536655;
        font-size: 12px;
    }

    .confidence-row strong {
        color: #326a40;
        font-size: 16px;
    }

    .confidence-track {
        height: 7px;
        margin-top: 12px;
        overflow: hidden;
        border-radius: 20px;
        background: #e6ede2;
    }

    .confidence-fill {
        height: 100%;
        border-radius: inherit;
        background: #63996b;
    }

    .disclaimer {
        margin: 24px 0;
        color: #5f7061;
        font-size: 11px;
        line-height: 1.8;
    }

    /* Footer */
    .footer {
        box-sizing: border-box;
        width: 84%;
        margin: 55px auto 0;
        padding: 22px 0 30px;
        display: flex;
        justify-content: space-between;
        gap: 15px;
        border-top: 1px solid #e0e7dc;
        background: transparent;
        color: #657466;
        font-size: 11px;
    }

    @media (max-width: 760px) {
        .navbar {
            padding: 16px 5%;
        }

        .brand {
            font-size: 14px;
        }

        .brand-icon {
            font-size: 21px;
        }

        .college-info strong {
            font-size: 11px;
        }

        .college-info span {
            font-size: 9px;
        }

        .hero {
            padding: 42px 18px 32px;
        }

        .hero h1 {
            letter-spacing: -1.5px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 16px;
        }

        .footer {
            width: 92%;
            font-size: 9px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL FUNCTIONS
# ============================================================
@st.cache_resource
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"Model file for '{model_name}' not found at: {path}. "
            "Please check that the file exists in your directory."
        )
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data
def load_class_names():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH.name}. "
            "Keep it in the same folder as app.py."
        )
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_class_name(classes, index):
    if isinstance(classes, list):
        if 0 <= index < len(classes):
            return str(classes[index])
    elif isinstance(classes, dict):
        if str(index) in classes:
            return str(classes[str(index)])
        if index in classes:
            return str(classes[index])

    raise ValueError(f"Class index {index} was not found in class_names.json.")


def predict_image(image, model_name: str):
    model = load_model(model_name)
    classes = load_class_names()

    resized = image.convert("RGB").resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    predictions = model.predict(array, verbose=0)
    scores = np.asarray(predictions[0])

    index = int(np.argmax(scores))
    predicted_class = get_class_name(classes, index)
    confidence = float(scores[index]) * 100.0

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_used": model_name,
    }


def predict_all_models(image):
    results = {}
    for name in MODEL_PATHS.keys():
        results[name] = predict_image(image, name)
    return results


# ============================================================
# SESSION STATE
# ============================================================
if "predictions_all" not in st.session_state:
    st.session_state.predictions_all = None
if "prediction_file" not in st.session_state:
    st.session_state.prediction_file = None
if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

# ============================================================
# HEADER
# ============================================================
st.html(
    """
    <div class="navbar">
        <div class="brand">
            <span class="brand-icon">✿</span>
            <span>Plant Disease Detection</span>
        </div>
        <div class="college-info">
            <strong>Aryan Dev Shridhar</strong>
            <span>USICT · GGSIPU</span>
        </div>
    </div>
    """
)

# ============================================================
# HERO
# ============================================================
st.html(
    """
    <div class="hero">
        <div class="eyebrow">
            <span class="status-dot"></span>
            DEEP LEARNING · PLANT HEALTH
        </div>
        <h1>A healthier garden<br>starts with <span>one leaf.</span></h1>
        <p class="subtitle">
            Upload a leaf photo to analyze and compare predictions across multiple
            deep learning architectures.
        </p>
    </div>
    """
)

# ============================================================
# TWO MAIN PANELS
# ============================================================
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.html(
            """
            <div class="section-heading">
                <div>
                    <div class="step-label">STEP 01</div>
                    <h2>Upload Image & Select Active Model</h2>
                </div>
                <span class="file-type">JPG · PNG · WEBP</span>
            </div>
            """
        )

        selected_model = st.selectbox(
            "Primary Model Selection",
            options=list(MODEL_PATHS.keys()),
            index=0,
            key="model_selector",
        )

        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"leaf_upload_{st.session_state.upload_version}",
        )

        image = None
        valid_image = False

        if uploaded is not None:
            if uploaded.size > MAX_SIZE:
                st.error("Image size must be less than 10 MB.")
                st.session_state.predictions_all = None
                st.session_state.prediction_file = None
            else:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    valid_image = True

                    if st.session_state.prediction_file != uploaded.name:
                        st.session_state.predictions_all = None
                        st.session_state.prediction_file = None

                    st.image(image, use_container_width=True)
                    st.caption(f"Selected file: {uploaded.name}")

                except (UnidentifiedImageError, OSError, ValueError):
                    st.error(
                        "Unable to read this image. Please upload a valid image."
                    )

        if st.button(
            "Detect Plant Disease →",
            disabled=not valid_image,
            key="predict_button",
        ):
            try:
                with st.spinner("Running predictions on all 3 models..."):
                    st.session_state.predictions_all = predict_all_models(image)
                    st.session_state.prediction_file = uploaded.name
            except Exception as error:
                st.session_state.predictions_all = None
                st.error(f"Prediction failed: {error}")

        st.html(
            """
            <p class="privacy-note">
                Image is processed locally across MobileNetV2, Custom CNN, and EfficientNetB0.
            </p>
            """
        )

        if uploaded is not None:
            if st.button("Remove image", key="remove_button"):
                st.session_state.predictions_all = None
                st.session_state.prediction_file = None
                st.session_state.upload_version += 1
                st.rerun()

with right:
    with st.container(border=True):
        st.html(
            """
            <div class="section-heading">
                <div>
                    <div class="step-label">STEP 02</div>
                    <h2>Analysis results</h2>
                </div>
                <span class="result-tag">MODEL OUTPUT</span>
            </div>
            """
        )

        all_preds = st.session_state.predictions_all

        if all_preds is None:
            st.html(
                """
                <div class="result-empty">
                    <div class="result-placeholder">01</div>
                    <strong>Your results will appear here</strong>
                    <p>Upload a leaf image and click detect to run inference.</p>
                    <div class="result-feature">
                        <span>✓</span>
                        38 plant disease and health categories
                    </div>
                    <div class="result-feature">
                        <span>✓</span>
                        Side-by-side comparison across all 3 models
                    </div>
                </div>
                """
            )
        else:
            view_mode = st.radio(
                "Display Mode",
                ["Selected Model", "Compare All 3 Models"],
                horizontal=True,
                label_visibility="collapsed",
                key="view_mode_toggle",
            )

            if view_mode == "Selected Model":
                res = all_preds[selected_model]
                predicted_class = (
                    str(res["predicted_class"])
                    .replace("___", " — ")
                    .replace("_", " ")
                )
                safe_class = html.escape(predicted_class)
                confidence = float(res["confidence"])
                bar_width = min(100.0, max(0.0, confidence))
                used_model_name = html.escape(str(res["model_used"]))

                st.html(
                    f"""
                    <div class="result-content">
                        <div class="result-symbol">✓</div>
                        <span class="result-overline">PREDICTED CLASS</span>
                        <div class="prediction-name">{safe_class}</div>
                        <div class="model-badge">Architecture: {used_model_name}</div>
                        <div class="confidence-row">
                            <span>Model confidence</span>
                            <strong>{confidence:.2f}%</strong>
                        </div>
                        <div class="confidence-track">
                            <div class="confidence-fill"
                                 style="width: {bar_width:.2f}%;"></div>
                        </div>
                        <p class="disclaimer">
                            This is a model prediction, not a confirmed diagnosis.
                            Accuracy may vary on real-world leaf photographs.
                        </p>
                    </div>
                    """
                )
            else:
                st.html("<div style='margin-top: 10px;'></div>")
                for m_name, res in all_preds.items():
                    pred_cls = (
                        str(res["predicted_class"])
                        .replace("___", " — ")
                        .replace("_", " ")
                    )
                    safe_cls = html.escape(pred_cls)
                    conf = float(res["confidence"])
                    bar_w = min(100.0, max(0.0, conf))

                    st.html(
                        f"""
                        <div class="compare-card">
                            <div class="compare-title">{m_name}</div>
                            <div class="compare-pred">{safe_cls}</div>
                            <div class="confidence-row">
                                <span>Confidence</span>
                                <strong>{conf:.2f}%</strong>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width: {bar_w:.2f}%;"></div>
                            </div>
                        </div>
                        """
                    )

            if st.button(
                "Analyze another image",
                key="another_image_button",
            ):
                st.session_state.predictions_all = None
                st.session_state.prediction_file = None
                st.session_state.upload_version += 1
                st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.html(
    """
    <div class="footer">
        <span>Plant Disease Detection Project</span>
        <span>Aryan Dev Shridhar · USICT, GGSIPU</span>
    </div>
    """
)