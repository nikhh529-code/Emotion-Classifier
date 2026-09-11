import streamlit as st
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Emotion Classifier",
    layout="centered"
)


@st.cache_resource
def load_artifacts():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        base_dir,
        "Artifacts",
        "BiGRU_Modle.keras"
    )

    tokenizer_path = os.path.join(
        base_dir,
        "Artifacts",
        "tokenizer.pkl"
    )

    if not os.path.exists(model_path):
        st.error(f"Model file not found:\n{model_path}")
        st.stop()

    if not os.path.exists(tokenizer_path):
        st.error(f"Tokenizer file not found:\n{tokenizer_path}")
        st.stop()

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


BiGRU, token = load_artifacts()

label_names = [
    "sadness", "joy", "love", "anger", "fear", "surprise"
]

st.title("Emotion Classifier")

st.write(
    "Enter a sentence and the trained BiGRU model "
    "will predict the emotion."
)


# Text input box
text = st.text_area(
    "Enter your text:",
    placeholder="Example: I feel so grateful today..."
)



if st.button("🔍 Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        sequence = token.texts_to_sequences([text])


        padded_sequence = pad_sequences(
            sequence,
            maxlen=66,
            padding="post",
            truncating="post"
        )

        prediction = BiGRU.predict(
            padded_sequence,
            verbose=0
        )


        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]


        predicted_emotion = label_names[predicted_class]


        confidence = prediction[0][predicted_class] * 100


        st.success(
            f"Predicted Emotion: **{predicted_emotion.upper()}**"
        )

        st.info(
            f"Confidence: **{confidence:.2f}%**"
        )


     