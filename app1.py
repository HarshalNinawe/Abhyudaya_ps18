import os
import PIL
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Constants
MODEL_PATHS = {
    "VGG16": "vgg16.model"
}

IMG_RES = {
    "resize": (128, 128),  # Update this size according to the model's expected input size if needed
    "input_shape": (128, 128, 3),
    "reshape": (-1, 128, 128, 3)
}

CLASSES = {
    0: "Actinic Keratoses And Intraepithelial Carcinomae",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis-Like Lesions",
    3: "Dermatofibroma",
    4: "Melanoma"
}

# Load the selected model
def load_selected_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if model_path:
        return load_model(model_path)
    else:
        raise ValueError("Invalid model name")

def predict(image, model):
    image = image.resize(IMG_RES["resize"])
    image = np.array(image).reshape(IMG_RES["reshape"]) / 255.0  # Normalize the image
    
    prediction = model.predict(image)[0]
    prediction = sorted(
        [(CLASSES[i], round(j * 100, 2)) for i, j in enumerate(prediction)],
        reverse=True,
        key=lambda x: x[1]
    )
    
    return prediction

# Streamlit app
def main():
    st.title("Skin Disease Detection")
    
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Upload an Image for Prediction")
        
        file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if file is not None:
            image = PIL.Image.open(file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            st.write("")
            st.subheader("Select Model for Prediction")
            model_choice = st.selectbox("Choose Model", sorted(MODEL_PATHS.keys()))
            
            if st.button("Classify"):
                st.write("Classifying...")
                model = load_selected_model(model_choice)
                prediction = predict(image, model)
                
                st.write(f"Model used: {model_choice}")
                for label, confidence in prediction:
                    st.write(f"{label}: {confidence}%")
    
    elif choice == "About":
        st.subheader("About Us")
        st.write("""

        """)

if __name__ == "__main__":
    main()
