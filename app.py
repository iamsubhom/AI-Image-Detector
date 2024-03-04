import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf 

#model=pickle.load(open("E:\\MyProjects\\\AI-Generated-Images-Classifier\\model\\modelai.pkl", "rb"))

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model_ai_classify.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()


# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title('AI vs Real Image Classification')
st.write('Sample Images to Test' )
st.image('real1.jpg', caption='Real Photo 1')
st.image('real2.jpg', caption='Real Photo 2')
st.image('ai1.jpg', caption='Ai Generated Photo 1')
st.image('ai2.jpg', caption='Ai Generated Photo 2')
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict(uploaded_image)
    if prediction[0][0] > 0.5:
        st.write('AI Generated Image')
    else:
        st.write('Real Image')

st.markdown('[Connect Me in LinkedIn](https://www.linkedin.com/in/iamsubhom)')
