
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection AI")

st.markdown(
    "### Smart agriculture powered by Artificial Intelligence"
)

st.write(
    "Upload a photograph of a plant leaf to identify "
    "its predicted disease."
)

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_model.keras"
    )

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names


model, class_names = load_model()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Your uploaded leaf",
        use_container_width=True
    )

    if st.button("🔍 Detect Disease", type="primary"):

        with st.spinner("AI is analysing your image..."):

            resized_image = image.resize((160, 160))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = preprocess_input(image_array)

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index]
            ) * 100

        if "___" in predicted_class:

            plant, disease = predicted_class.split("___", 1)

        else:

            plant = "Unknown"
            disease = predicted_class

        plant = plant.replace("_", " ")
        disease = disease.replace("_", " ")

        st.success("Analysis completed!")

        st.subheader("🌱 Prediction Results")

        st.write("**Plant:**", plant)

        st.write("**Predicted Disease:**", disease)

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(predictions)[-3:][::-1]

        for index in top_indices:

            label = class_names[index].replace("___", " - ")

            score = float(predictions[index])

            st.write(f"{label}: {score * 100:.2f}%")

            st.progress(score)

        st.info(
            "This is an AI prediction, not a confirmed "
            "agricultural diagnosis. Results on real-world "
            "images may differ from PlantVillage results."
        )

else:

    st.info("Upload a leaf photograph to start.")
