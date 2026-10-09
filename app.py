import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroVision AI | Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(112, 64, 255, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(0, 220, 255, 0.12),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #050817 0%,
            #080d1f 50%,
            #050817 100%
        );
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0c122a,
            #060a19
        );

    border-right: 1px solid rgba(148,163,184,0.12);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] {
    background: #111a31 !important;
    border: 2px dashed #475569 !important;
    border-radius: 18px !important;
    min-height: 190px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #e2e8f0 !important;
}

[data-testid="stFileUploaderDropzone"] small {
    color: #94a3b8 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #0891b2
    ) !important;

    color: #ffffff !important;

    border: none !important;
    border-radius: 10px !important;

    font-weight: 700 !important;

    padding: 8px 18px !important;
}

[data-testid="stFileUploaderDropzone"] button span {
    color: #ffffff !important;
}

[data-testid="stFileUploaderFile"] {
    background: #111a31 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #e2e8f0 !important;
}


/* ============================================================
   NORMAL STREAMLIT BUTTONS
   ============================================================ */

.stButton > button {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #0891b2
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PATH = "brain_tumor_efficientnetb0.keras"
CLASS_PATH = "class_names.json"

EXPECTED_CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_classes():

    try:

        with open(CLASS_PATH, "r") as f:
            classes = json.load(f)

        return classes

    except Exception:

        return EXPECTED_CLASSES


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:

    model = load_model()
    class_names = load_classes()

except Exception as e:

    st.error(
        "⚠️ Model files could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="
        padding:10px 5px 25px 5px;
    ">

        <div style="
            width:54px;
            height:54px;
            border-radius:16px;
            background:linear-gradient(
                135deg,
                #7c3aed,
                #06b6d4
            );
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:27px;
            margin-bottom:15px;
            box-shadow:
                0 10px 30px rgba(124,58,237,0.3);
        ">
            🧠
        </div>

        <div style="
            font-size:21px;
            font-weight:800;
            color:#ffffff;
        ">
            NeuroVision AI
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            margin-top:4px;
        ">
            Brain MRI Intelligence Platform
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        color:#64748b;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.5px;
        margin:15px 0 10px 0;
    ">
        MODEL
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.035);
        border:1px solid rgba(148,163,184,0.12);
        border-radius:14px;
        padding:15px;
        margin-bottom:10px;
    ">

        <div style="
            color:#ffffff;
            font-size:13px;
            font-weight:700;
            margin-bottom:6px;
        ">
            EfficientNetB0
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            line-height:1.6;
        ">
            Transfer-learning based deep neural network
            for multi-class MRI image classification.
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        color:#64748b;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.5px;
        margin:25px 0 10px 0;
    ">
        INPUT
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.035);
        border:1px solid rgba(148,163,184,0.12);
        border-radius:14px;
        padding:15px;
    ">

        <div style="
            color:#ffffff;
            font-size:13px;
            font-weight:700;
            margin-bottom:6px;
        ">
            MRI Image
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            line-height:1.6;
        ">
            JPG, JPEG and PNG images<br>
            Input size: 224 × 224
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        color:#64748b;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.5px;
        margin:25px 0 10px 0;
    ">
        CLASSES
    </div>
    """, unsafe_allow_html=True)


    for cls in class_names:

        display_name = cls.replace(
            "_",
            " "
        ).title()

        st.markdown(
            f"""
            <div style="
                padding:8px 10px;
                margin-bottom:5px;
                border-radius:8px;
                background:rgba(255,255,255,0.035);
                color:#cbd5e1;
                font-size:12px;
            ">
                <span style="color:#67e8f9;">●</span>
                {display_name}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("""
    <div style="
        color:#64748b;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.5px;
        margin:25px 0 10px 0;
    ">
        SYSTEM
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.035);
        border:1px solid rgba(148,163,184,0.12);
        border-radius:14px;
        padding:15px;
    ">

        <div style="
            color:#ffffff;
            font-size:13px;
            font-weight:700;
            margin-bottom:8px;
        ">
            System Status
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            line-height:1.8;
        ">
            <span style="color:#22c55e;">●</span>
            Model loaded<br>

            <span style="color:#22c55e;">●</span>
            Inference ready<br>

            <span style="color:#22c55e;">●</span>
            System online
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.html("""
<div style="
    position:relative;
    overflow:hidden;
    padding:45px 50px;
    border-radius:24px;

    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(6,182,212,0.14),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(124,58,237,0.18),
            transparent 30%
        ),
        #0f172a;

    border:1px solid rgba(148,163,184,0.15);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.35);

    margin-bottom:35px;
">

    <div style="
        position:absolute;
        top:0;
        left:0;
        width:5px;
        height:100%;
        background:
            linear-gradient(
                180deg,
                #8b5cf6,
                #06b6d4
            );
    "></div>

    <div style="
        display:inline-block;
        padding:8px 15px;
        border-radius:50px;

        background:rgba(124,58,237,0.14);
        border:1px solid rgba(139,92,246,0.35);

        color:#c4b5fd;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.2px;

        margin-bottom:18px;
    ">
        ✦ AI-POWERED MEDICAL IMAGE CLASSIFICATION
    </div>

    <div style="
        font-size:58px;
        line-height:1.05;
        font-weight:800;
        color:#ffffff;
        margin-bottom:18px;
    ">
        Brain MRI Intelligence
    </div>

    <div style="
        max-width:850px;
        color:#94a3b8;
        font-size:15px;
        line-height:1.8;
    ">
        An intelligent deep-learning dashboard for classifying
        brain MRI images using an EfficientNetB0 transfer-learning model.
        Upload an MRI scan and explore the model's predicted class,
        confidence score, and probability distribution.
    </div>

</div>
""")


# ============================================================
# PERFORMANCE TITLE
# ============================================================

st.html("""
<div style="
    font-size:27px;
    font-weight:800;
    color:#f8fafc;
    margin-top:10px;
    margin-bottom:6px;
">
    Model Performance
</div>

<div style="
    color:#64748b;
    font-size:13px;
    margin-bottom:22px;
">
    Evaluation metrics obtained from the held-out internal test split.
</div>
""")


# ============================================================
# PERFORMANCE CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html("""
    <div style="
        background:
            linear-gradient(
                145deg,
                rgba(30,41,72,0.9),
                rgba(15,23,42,0.95)
            );

        border:1px solid rgba(139,92,246,0.22);
        border-radius:18px;
        padding:23px;
        min-height:130px;
        box-shadow:
            0 15px 40px rgba(0,0,0,0.20);
    ">

        <div style="
            color:#94a3b8;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            TEST ACCURACY
        </div>

        <div style="
            color:#c4b5fd;
            font-size:32px;
            font-weight:800;
            margin-top:10px;
        ">
            59.46%
        </div>

        <div style="
            color:#64748b;
            font-size:11px;
            margin-top:5px;
        ">
            Classification accuracy
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div style="
        background:
            linear-gradient(
                145deg,
                rgba(30,41,72,0.9),
                rgba(15,23,42,0.95)
            );

        border:1px solid rgba(103,232,249,0.18);
        border-radius:18px;
        padding:23px;
        min-height:130px;
    ">

        <div style="
            color:#94a3b8;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            PRECISION
        </div>

        <div style="
            color:#67e8f9;
            font-size:32px;
            font-weight:800;
            margin-top:10px;
        ">
            61.82%
        </div>

        <div style="
            color:#64748b;
            font-size:11px;
            margin-top:5px;
        ">
            Weighted precision
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div style="
        background:
            linear-gradient(
                145deg,
                rgba(30,41,72,0.9),
                rgba(15,23,42,0.95)
            );

        border:1px solid rgba(103,232,249,0.18);
        border-radius:18px;
        padding:23px;
        min-height:130px;
    ">

        <div style="
            color:#94a3b8;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            RECALL
        </div>

        <div style="
            color:#67e8f9;
            font-size:32px;
            font-weight:800;
            margin-top:10px;
        ">
            59.46%
        </div>

        <div style="
            color:#64748b;
            font-size:11px;
            margin-top:5px;
        ">
            Weighted recall
        </div>

    </div>
    """)


with col4:

    st.html("""
    <div style="
        background:
            linear-gradient(
                145deg,
                rgba(30,41,72,0.9),
                rgba(15,23,42,0.95)
            );

        border:1px solid rgba(103,232,249,0.18);
        border-radius:18px;
        padding:23px;
        min-height:130px;
    ">

        <div style="
            color:#94a3b8;
            font-size:11px;
            font-weight:700;
            letter-spacing:1.2px;
        ">
            F1 SCORE
        </div>

        <div style="
            color:#67e8f9;
            font-size:32px;
            font-weight:800;
            margin-top:10px;
        ">
            59.62%
        </div>

        <div style="
            color:#64748b;
            font-size:11px;
            margin-top:5px;
        ">
            Weighted F1 score
        </div>

    </div>
    """)


# ============================================================
# MRI ANALYSIS
# ============================================================

st.html("""
<div style="
    font-size:27px;
    font-weight:800;
    color:#f8fafc;
    margin-top:38px;
    margin-bottom:6px;
">
    MRI Analysis
</div>

<div style="
    color:#64748b;
    font-size:13px;
    margin-bottom:22px;
">
    Upload an MRI image to run AI-assisted classification.
</div>
""")


# ============================================================
# UPLOAD CONTAINER
# ============================================================

st.html("""
<div style="
    background:
        linear-gradient(
            145deg,
            rgba(17,25,48,0.95),
            rgba(10,15,32,0.98)
        );

    border:1px solid rgba(139,92,246,0.20);
    border-radius:22px;
    padding:24px;

    margin-bottom:10px;
">

    <div style="
        color:#f8fafc;
        font-size:15px;
        font-weight:700;
        margin-bottom:12px;
    ">
        🧬 Upload Brain MRI Scan
    </div>

</div>
""")


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(
        [1, 1.15],
        gap="large"
    )


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    with left:

        st.html("""
        <div style="
            background:#0f172a;
            border:1px solid rgba(148,163,184,0.14);
            border-radius:20px;
            padding:20px;
            box-shadow:
                0 18px 50px rgba(0,0,0,0.25);
        ">

            <div style="
                color:#f8fafc;
                font-size:15px;
                font-weight:700;
                margin-bottom:15px;
            ">
                🖼️ MRI Scan Preview
            </div>

        </div>
        """)

        st.image(
            image,
            use_container_width=True
        )


    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    image_resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image_resized
    ).astype("float32")

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with st.spinner(
        "Analyzing MRI image..."
    ):

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        float(
            predictions[predicted_index]
        ) * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    with right:

        display_prediction = (
            predicted_class
            .replace("_", " ")
            .title()
        )


        st.html(f"""
        <div style="
            background:
                radial-gradient(
                    circle at 90% 10%,
                    rgba(6,182,212,0.12),
                    transparent 35%
                ),
                linear-gradient(
                    145deg,
                    rgba(30,41,72,0.94),
                    rgba(15,23,42,0.96)
                );

            border:1px solid rgba(103,232,249,0.20);
            border-radius:22px;

            padding:28px;

            box-shadow:
                0 20px 55px rgba(0,0,0,0.25);
        ">

            <div style="
                color:#94a3b8;
                font-size:11px;
                font-weight:700;
                letter-spacing:1.4px;
            ">
                AI PREDICTION
            </div>

            <div style="
                color:#ffffff;
                font-size:35px;
                font-weight:800;
                margin:8px 0 12px 0;
            ">
                {display_prediction}
            </div>

            <div style="
                color:#94a3b8;
                font-size:11px;
                font-weight:700;
                letter-spacing:1.4px;
            ">
                CONFIDENCE SCORE
            </div>

            <div style="
                color:#67e8f9;
                font-size:28px;
                font-weight:800;
                margin-top:6px;
            ">
                {confidence:.2f}%
            </div>

        </div>
        """)


        st.html("""
        <div style="
            color:#64748b;
            font-size:13px;
            margin-top:22px;
            margin-bottom:12px;
        ">
            Probability distribution across all classes
        </div>
        """)


        # ====================================================
        # PROBABILITY BARS
        # ====================================================

        for i, cls in enumerate(class_names):

            probability = (
                float(
                    predictions[i]
                ) * 100
            )

            display_name = (
                cls
                .replace("_", " ")
                .title()
            )


            st.html(f"""
            <div style="
                background:rgba(15,23,42,0.82);
                border:1px solid rgba(148,163,184,0.12);
                border-radius:16px;
                padding:17px;
                margin-bottom:10px;
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:8px;
                ">

                    <span style="
                        color:#e2e8f0;
                        font-size:13px;
                        font-weight:600;
                    ">
                        {display_name}
                    </span>

                    <span style="
                        color:#67e8f9;
                        font-size:13px;
                        font-weight:700;
                    ">
                        {probability:.2f}%
                    </span>

                </div>

                <div style="
                    height:7px;
                    width:100%;
                    background:#1e293b;
                    border-radius:20px;
                    overflow:hidden;
                ">

                    <div style="
                        height:100%;
                        width:{probability:.2f}%;
                        border-radius:20px;
                        background:
                            linear-gradient(
                                90deg,
                                #7c3aed,
                                #06b6d4
                            );
                    "></div>

                </div>

            </div>
            """)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.html("""
<div style="
    font-size:27px;
    font-weight:800;
    color:#f8fafc;
    margin-top:42px;
    margin-bottom:6px;
">
    Model Architecture
</div>

<div style="
    color:#64748b;
    font-size:13px;
    margin-bottom:22px;
">
    Key components used to build the classification pipeline.
</div>
""")


info1, info2, info3 = st.columns(3)


with info1:

    st.html("""
    <div style="
        height:100%;
        padding:22px;

        background:#0f172a;

        border:1px solid rgba(148,163,184,0.12);
        border-radius:18px;
    ">

        <div style="
            font-size:25px;
            margin-bottom:12px;
        ">
            🧠
        </div>

        <div style="
            color:#ffffff;
            font-size:14px;
            font-weight:700;
            margin-bottom:7px;
        ">
            Transfer Learning
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            line-height:1.7;
        ">
            EfficientNetB0 uses ImageNet-pretrained
            feature representations to improve image
            classification performance with limited data.
        </div>

    </div>
    """)


with info2:

    st.html("""
    <div style="
        height:100%;
        padding:22px;

        background:#0f172a;

        border:1px solid rgba(148,163,184,0.12);
        border-radius:18px;
    ">

        <div style="
            font-size:25px;
            margin-bottom:12px;
        ">
            🔬
        </div>

        <div style="
            color:#ffffff;
            font-size:14px;
            font-weight:700;
            margin-bottom:7px;
        ">
            Image Processing
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            line-height:1.7;
        ">
            Uploaded MRI images are converted to RGB
            format and resized to 224 × 224 pixels
            before model inference.
        </div>

    </div>
    """)


with info3:

    st.html("""
    <div style="
        height:100%;
        padding:22px;

        background:#0f172a;

        border:1px solid rgba(148,163,184,0.12);
        border-radius:18px;
    ">

        <div style="
            font-size:25px;
            margin-bottom:12px;
        ">
            📊
        </div>

        <div style="
            color:#ffffff;
            font-size:14px;
            font-weight:700;
            margin-bottom:7px;
        ">
            Multi-Class Prediction
        </div>

        <div style="
            color:#94a3b8;
            font-size:12px;
            line-height:1.7;
        ">
            The model produces probability scores for
            glioma, meningioma, no tumor and pituitary
            classes.
        </div>

    </div>
    """)


# ============================================================
# DISCLAIMER
# ============================================================

st.html("""
<div style="
    background:rgba(120,53,15,0.12);
    border:1px solid rgba(251,191,36,0.20);
    border-radius:16px;

    padding:17px 20px;

    color:#fcd34d;
    font-size:12px;
    line-height:1.7;

    margin-top:30px;
">
    ⚠️ <strong>Medical AI Disclaimer:</strong>
    This application is developed for academic and demonstration
    purposes. The predictions are AI-assisted image classification
    results and should not be considered a medical diagnosis or a
    substitute for professional clinical evaluation.
</div>
""")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div style="
    text-align:center;
    color:#475569;
    font-size:11px;

    margin-top:45px;
    padding-top:20px;

    border-top:
        1px solid rgba(148,163,184,0.08);
">

    NeuroVision AI • Brain MRI Classification Dashboard

    <br>

    EfficientNetB0 • Transfer Learning • TensorFlow • Streamlit

</div>
""")