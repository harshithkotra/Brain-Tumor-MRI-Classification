import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain MRI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(124, 58, 237, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6, 182, 212, 0.12),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #050817 0%,
                #080b1c 50%,
                #050817 100%
            );
    }


    .main .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       STREAMLIT HEADER
       ======================================================== */

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       ALL TEXT VISIBILITY
       ======================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown li,
    .stMarkdown strong {
        color: #cbd5e1 !important;
    }


    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }


    label {
        color: #cbd5e1 !important;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: #ffffff !important;
        font-weight: 900 !important;
        letter-spacing: -1px !important;
    }


    h2 {
        color: #ffffff !important;
        font-weight: 850 !important;
    }


    h3 {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080b1c 0%,
                #070a18 100%
            );

        border-right:
            1px solid
            rgba(124, 58, 237, 0.25);
    }


    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {
        color: #cbd5e1 !important;
    }


    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }


    section[data-testid="stSidebar"] .stAlert {
        background:
            rgba(30, 41, 70, 0.90);

        border:
            1px solid
            rgba(96, 165, 250, 0.18);

        border-radius: 16px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 70, 0.92),
                rgba(15, 23, 42, 0.96)
            );

        border:
            1px solid
            rgba(96, 165, 250, 0.16);

        border-radius: 20px;

        padding: 22px;

        box-shadow:
            0 15px 40px
            rgba(0, 0, 0, 0.22);
    }


    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }


    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }


    div[data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;

        border-radius: 14px;

        border:
            1px solid
            rgba(139, 92, 246, 0.35);

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #06b6d4
            );

        color: white !important;

        font-weight: 800;

        min-height: 45px;
    }


    .stButton > button:hover {
        border-color: #a78bfa;

        box-shadow:
            0 8px 25px
            rgba(124, 58, 237, 0.25);
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background:
            rgba(15, 23, 42, 0.65);

        border-radius: 18px;

        border:
            1px solid
            rgba(96, 165, 250, 0.14);
    }


    [data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
    }


    [data-testid="stFileUploader"] small {
        color: #94a3b8 !important;
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    .stProgress > div > div > div > div {
        background:
            linear-gradient(
                90deg,
                #7c3aed,
                #06b6d4
            );
    }


    /* ========================================================
       IMAGE
       ======================================================== */

    img {
        border-radius: 18px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color:
            rgba(148, 163, 184, 0.12) !important;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_brain_model():

    return load_model(
        "brain_tumor_efficientnetb0.keras",
        compile=False
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(
        "class_names.json",
        "r"
    ) as file:

        data = json.load(file)


    if isinstance(data, list):

        return data


    if isinstance(data, dict):

        if all(
            str(i) in data
            for i in range(len(data))
        ):

            return [
                data[str(i)]
                for i in range(len(data))
            ]

        return list(data.values())


    return [
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary"
    ]


# ============================================================
# LOAD MODEL + LABELS
# ============================================================

try:

    model = load_brain_model()

    class_names = load_class_names()

except Exception as error:

    st.error(
        "❌ Model files could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🧠"
    )

    st.markdown(
        "## NeuroVision AI"
    )

    st.caption(
        "Brain MRI Intelligence"
    )

    st.divider()


    st.markdown(
        "### ⚙️ SYSTEM"
    )


    st.info(
        "🧠 **Model**\n\n"
        "EfficientNetB0 Transfer Learning"
    )


    st.info(
        "🖼️ **Input**\n\n"
        "Brain MRI image\n\n"
        "224 × 224 pixels"
    )


    st.info(
        "🔬 **Classes**\n\n"
        "Glioma\n\n"
        "Meningioma\n\n"
        "No Tumor\n\n"
        "Pituitary"
    )


    st.success(
        "🟢 Model loaded successfully"
    )


    st.divider()


    st.markdown(
        "### ⚕️ ABOUT"
    )


    st.caption(
        "This application is an AI-assisted "
        "research and educational tool. "
        "It is not a replacement for "
        "professional medical diagnosis."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    "### ✦ AI-POWERED MEDICAL IMAGE CLASSIFICATION"
)


st.title(
    "Brain MRI Intelligence"
)


st.write(
    "An intelligent deep-learning dashboard for "
    "classifying brain MRI images using an "
    "EfficientNetB0 transfer-learning model. "
    "Upload an MRI scan and explore the model's "
    "predicted class, confidence score, and "
    "probability distribution."
)


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header(
    "Model Performance"
)


st.caption(
    "Evaluation metrics obtained from the "
    "held-out internal test split."
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "TEST ACCURACY",
        "59.46%"
    )


with metric2:

    st.metric(
        "PRECISION",
        "61.82%"
    )


with metric3:

    st.metric(
        "RECALL",
        "59.46%"
    )


with metric4:

    st.metric(
        "F1 SCORE",
        "59.62%"
    )


st.markdown("")


# ============================================================
# MODEL DETAILS
# ============================================================

details_left, details_right = st.columns(
    2,
    gap="large"
)


# ------------------------------------------------------------
# ARCHITECTURE
# ------------------------------------------------------------

with details_left:

    st.subheader(
        "🧠 EfficientNetB0 Architecture"
    )


    st.write(
        "Transfer-learning based image classification "
        "pipeline using an ImageNet-pretrained "
        "EfficientNetB0."
    )


    st.markdown(
        """
        **Backbone:** EfficientNetB0

        **Pretraining:** ImageNet

        **Input:** 224 × 224 × 3 RGB image

        **Classification Head:** Global Average
        Pooling, Batch Normalization, Dense and Dropout

        **Output:** 4-class Softmax

        **Total Parameters:** 4,219,175

        **Trainable Parameters:** 167,044
        """
    )


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

with details_right:

    st.subheader(
        "📊 Dataset & Classes"
    )


    st.write(
        "The model classifies brain MRI images "
        "into four categories."
    )


    st.markdown(
        """
        🧠 **Glioma**

        🧠 **Meningioma**

        🧠 **No Tumor**

        🧠 **Pituitary**
        """
    )


    st.info(
        "The reported metrics are from the "
        "held-out internal test split."
    )


st.divider()


# ============================================================
# MRI CLASSIFICATION
# ============================================================

st.header(
    "MRI Image Classification"
)


st.caption(
    "Upload a brain MRI image to generate "
    "an AI-assisted classification."
)


uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        image_col, result_col = st.columns(
            [1, 1.2],
            gap="large"
        )


        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        with image_col:

            st.subheader(
                "🖼️ Uploaded MRI"
            )


            st.image(
                image,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        resized_image = image.resize(
            (224, 224)
        )


        image_array = np.array(
            resized_image
        ).astype("float32")


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(
                predictions
            )
        )


        predicted_class = class_names[
            predicted_index
        ]


        confidence = (
            float(
                predictions[
                    predicted_index
                ]
            ) * 100
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with result_col:

            st.subheader(
                "🎯 Prediction Result"
            )


            st.success(
                "Predicted Class: "
                + predicted_class.replace(
                    "_",
                    " "
                ).title()
            )


            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            st.markdown(
                "### 📈 Probability Distribution"
            )


            for index, name in enumerate(
                class_names
            ):

                probability = (
                    float(
                        predictions[index]
                    ) * 100
                )


                st.write(
                    name.replace(
                        "_",
                        " "
                    ).title()
                )


                st.progress(
                    min(
                        probability / 100,
                        1.0
                    ),
                    text=f"{probability:.2f}%"
                )


    except Exception as error:

        st.error(
            "❌ Unable to process the uploaded image."
        )


        st.code(
            str(error)
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "🧠 Upload an MRI scan above "
        "to begin classification."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ Medical disclaimer: This application is "
    "an AI-assisted research and educational tool "
    "for brain MRI image classification. Its "
    "predictions should not be considered a medical "
    "diagnosis or a replacement for evaluation by "
    "a qualified healthcare professional."
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Brain MRI Intelligence • EfficientNetB0 "
    "Transfer Learning • TensorFlow/Keras • Streamlit"
)