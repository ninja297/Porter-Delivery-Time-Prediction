
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json

model = tf.keras.models.load_model("porter_delivery_model.keras")
preprocessor = joblib.load("preprocessor.pkl")

with open("schema.json", "r") as f:
    schema = json.load(f)

st.set_page_config(
    page_title="Porter Delivery Predictor",
    page_icon="🚚"
)

st.title("🚚 Porter Delivery Time Predictor")
st.write("Predict estimated food delivery time using AI.")

user_input = {}

with st.form("prediction_form"):

    st.subheader("Enter Order Details")

    for col, default in schema["numeric"].items():
        user_input[col] = st.number_input(
            col.replace("_", " ").title(),
            value=float(default)
        )

    for col, options in schema["categorical"].items():
        choices = options if options else ["Unknown"]

        user_input[col] = st.selectbox(
            col.replace("_", " ").title(),
            choices
        )

    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:

    input_df = pd.DataFrame([user_input])

    processed_input = preprocessor.transform(input_df)

    processed_input = np.asarray(
        processed_input,
        dtype=np.float32
    )

    prediction = model.predict(
        processed_input,
        verbose=0
    )[0][0]

    st.success(
        f"Estimated Delivery Time: {max(0, prediction):.2f} minutes"
    )
