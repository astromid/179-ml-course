import dill
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


def app():
    if "model" not in st.session_state:
        st.session_state["model"] = None

    if st.session_state.model is None:
        model_checkpoint = st.file_uploader("Upload model checkpoint", type=["dill"])
        if model_checkpoint is not None:
            st.session_state.model = dill.load(model_checkpoint)
            st.success("Model loaded")

    if st.session_state.model is not None:
        canvas = st_canvas(background_color="black", stroke_color="white", height=280, width=280)
        image_array = canvas.image_data[:, :, :-1]
        if image_array is not None and st.button("Predict"):
            image = Image.fromarray(image_array).resize((28, 28), resample=Image.LANCZOS).convert('L')
            sample = np.asarray(image).flatten() / 255
            st.image(sample.reshape(28, 28), caption="Input image", width=100)

            scores = st.session_state.model.forward(sample.reshape(1, -1))
            st.write("Probaility distribution: ", softmax(scores))
            st.write("Predicted class: ", np.argmax(scores))


def softmax(scores):
    shifted_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
    exp_scores = np.exp(shifted_scores)
    sum_scores = np.sum(exp_scores, axis=1).reshape(-1, 1)
    return exp_scores / sum_scores


if __name__ == "__main__":
    app()
