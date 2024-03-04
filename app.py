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
image_paths = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg",
    "image4.jpg"
]

# Display images in a grid
image_paths = [
    "real1.jpg",
    "real2.jpg",
    "Ai1.jpg",
    "Ai2.jpg"
]
col1, col2 = st.columns(2)
with col1:
    st.image(image_paths[0], use_column_width=True)
with col2:
    st.image(image_paths[1], use_column_width=True)
col3, col4 = st.columns(2)
with col3:
    st.image(image_paths[2], use_column_width=True)
with col4:
    st.image(image_paths[3], use_column_width=True)
# Create a grid layout using HTML/CSS
st.write(
    """
    <style>
    .grid-container {
      display: grid;
      grid-template-columns: auto auto;
      gap: 10px;
    }
    .grid-item {
      padding: 10px;
    }
    </style>
    """
)

# Display images in the grid layout
st.write('<div class="grid-container">', unsafe_allow_html=True)
for image_path in image_paths:
    st.write('<div class="grid-item">', unsafe_allow_html=True)
    st.image(image_path, use_column_width=True)
    st.write('</div>', unsafe_allow_html=True)
st.write('</div>', unsafe_allow_html=True)



st.title('AI vs Real Image Classification')
#st.write('Sample Images to Test' )
#st.image('real1.jpg', caption='Real Photo 1')
#st.image('real2.jpg', caption='Real Photo 2')
#st.image('Ai1.jpg', caption='Ai Generated Photo 1')
#st.image('Ai2.jpg', caption='Ai Generated Photo 2')
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
