import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st

# 1. Load the model
# NOTE: Ensure 'brain_tumor_model.h5' matches the actual filename of your saved model.
# If your model is in a folder, provide the full path e.g., 'path/to/model.h5'
@st.cache_resource
def get_model():
    # Replace 'brain_tumor_model.h5' with your actual model file name
    model = load_model('brain_tumor_model.h5')
    return model

try:
    model = get_model()
except OSError:
    st.error("Model file not found. Please ensure the model file is saved in the same directory and named correctly (e.g., 'brain_tumor_model.h5').")
    st.stop()

# 2. Display Streamlit page title and prompt
st.title('Brain Tumor Detection System')
st.write("Upload a Brain MRI image below, and the CNN model will classify the tumor type.")

# 3. File uploader
image_input = st.file_uploader('Upload your MRI image here:', type=['jpg', 'jpeg', 'png'])

# 4. Classify button and logic
button = st.button('Classify Image')

if image_input is not None:
    # Display the uploaded image
    st.image(image_input, caption='Uploaded MRI', use_container_width=True)

if button and image_input is not None:
    # --- Preprocessing steps following your project settings ---
    
    # Load image with target size (150, 150) as used in your notebook
    img = load_img(image_input, target_size=(150, 150))
    
    # Convert to array
    img = img_to_array(img)
    
    # Expand dimensions to match model input shape (1, 150, 150, 3)
    img = np.expand_dims(img, axis=0)
    
    # Normalize pixel values (rescale=1./255) using ImageDataGenerator
    # Note: Your notebook used rescale=1./255, so we apply that here.
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow(x=img, batch_size=1)
    
    # Make prediction
    pred = model.predict(generator)
    class_index = np.argmax(pred, axis=1)[0]
    
    # --- Result Display (following the pattern of the sample file) ---
    st.subheader('Prediction Result:')

    # Class mappings from your notebook: 
    # {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    
    if class_index == 0:
        st.write('### The model predicts: **Glioma**')
        st.info("Glioma is a type of tumor that occurs in the brain and spinal cord.")
    elif class_index == 1:
        st.write('### The model predicts: **Meningioma**')
        st.info("Meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull.")
    elif class_index == 2:
        st.write('### The model predicts: **No Tumor**')
        st.success("Great news! The model did not detect a brain tumor in this image.")
    elif class_index == 3:
        st.write('### The model predicts: **Pituitary**')
        st.info("Pituitary tumors are abnormal growths that develop in your pituitary gland.")
    
    # Optional: Show confidence scores
    # st.write(f"Confidence: {np.max(pred) * 100:.2f}%")

elif button and image_input is None:
    st.warning("Please upload an image first.")