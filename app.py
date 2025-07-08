import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# loading trained model
model = tf.keras.models.load_model('skin_disease_model.h5')

# Call model once to build it
dummy_input = tf.zeros((1,224,224,3))
_ = model(dummy_input)

class_names = ['melanoma', 'nevus', 'seborrheic_keratosis']

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e1e1e, #3a3a3a);
    color: white;
}
div.stButton > button {
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
}
div.stFileUploader, div[data-testid="stCameraInput"], div[data-testid="stSelectbox"] {
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Skin Disease Classifier")

option = st.selectbox(
    "Choose input method:",
    ("Select an option", "Upload Image", "Capture Image")
)

image = None

# Image options
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

elif option == "Capture Image":
    captured_image = st.camera_input("Take a photo")
    if captured_image is not None:
        image = Image.open(captured_image).convert('RGB')

if image:
    st.image(image, caption='Selected Image', use_column_width=True)

    if st.button('Predict'):
        img_resized = image.resize((224,224))
        img_array = np.array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.write(f"### 🩺 Predicted Disease: **{predicted_class}**")
        st.write(f"Confidence: {confidence:.2f}")
