from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "digit_recognizer.keras"

st.set_page_config(page_title="NeuroDigits", page_icon="✍️", layout="wide")
st.title("✍️ NeuroDigits")
st.caption("Handwritten Digit Recognition · Feedforward Neural Network")
st.write("Draw a digit on the canvas and get the predicted digit with its confidence percentage. No graphs or image uploads.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
if model is None:
    st.warning("The trained model was not found.")
    st.write("Run setup.bat once, then train.bat once. After the model is created, use run.bat whenever you want to open NeuroDigits.")
    st.stop()


def prepare_canvas_image(image: Image.Image) -> np.ndarray:
    """Prepare the white-on-black drawing as 784 normalized pixel values."""
    image = image.convert("L")
    arr = np.asarray(image)
    # Canvas should be black with a white digit. Reject an empty drawing.
    mask = arr > 25
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 784), dtype=np.float32)

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    crop = image.crop((left, top, right, bottom))
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.25))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    return pixels.reshape(1, 784)


st.subheader("Draw your digit")
st.caption("Use your mouse to draw one large digit in white on the black canvas.")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=16,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="digit_canvas",
)

if st.button("Predict drawn digit", type="primary", key="predict_draw"):
    if canvas_result.image_data is None:
        st.error("Please draw a digit first.")
    else:
        rgba = np.asarray(canvas_result.image_data, dtype=np.uint8)
        drawn = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        prepared = prepare_canvas_image(drawn)
        if not np.any(prepared):
            st.error("No digit detected. Please draw a digit on the canvas first.")
        else:
            probabilities = model.predict(prepared, verbose=0)[0]
            digit = int(np.argmax(probabilities))
            confidence = float(probabilities[digit]) * 100
            st.subheader("Prediction")
            left, right = st.columns(2)
            with left:
                st.metric("Predicted digit", str(digit))
            with right:
                st.metric("Confidence", f"{confidence:.2f}%")
            st.caption("Confidence is the model's softmax score, not a guarantee that the prediction is correct.")
            st.write("Processed 28 × 28 input")
            preview = Image.fromarray((prepared.reshape(28, 28) * 255).astype(np.uint8))
            st.image(preview, width=180)

with st.expander("About this model"):
    st.write("NeuroDigits uses a feedforward neural network trained on the labelled train.csv file from the Kaggle Digit Recognizer dataset. The model is not retrained when you make a prediction.")
