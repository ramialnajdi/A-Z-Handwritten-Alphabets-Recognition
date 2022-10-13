import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import string 
import time

st.set_page_config(
page_title = "Handwritten Letters Classifier",
page_icon = ":pencil:",
)

hide_streamlit_style = """            
                       <style>            
                       #MainMenu {visibility: hidden;}            
                       footer {visibility: hidden;}            
                       </style>            
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("A-Z Handwritten Alphabets Recognition App")
st.markdown("This app Recognise the **Handwritten Alphabets**!")

st.markdown("1-paint a letter.")
st.markdown("2-click on the left bottom button.")
st.markdown("3-clik on the predict butten.")

st.sidebar.header("Configuration")

# Specify brush parameters and drawing mode.
b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)

# Create a canvas component
canvas_result = st_canvas(
            fill_color="#eee",
            stroke_width=b_width,
            stroke_color="white",
            background_color="black",
            update_streamlit=False,
            height=500,
            width=500,
            drawing_mode="freedraw",
        )


predict = st.button("Predict")

word_dict = dict() # make dictionary contains all english letters.
for i, char in enumerate(string.ascii_uppercase):
    word_dict[i] = char

def get_prediction(image):
    # get the prediction from your model and return it
    path = "web_app/model_hand.h5" # for example.
    CNN_model = tf.keras.models.load_model(path) # import trained model.
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im_gray  =cv2.GaussianBlur(im_gray, (15,15), 0)
    ret, im_th = cv2.threshold(im_gray,100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28,28), interpolation=cv2.INTER_AREA)
    X_re = np.reshape(roi, (1, 28, 28, 1))
    predictions = word_dict[np.argmax(CNN_model.predict(X_re))]
    return predictions 

if canvas_result.image_data is not None and predict:
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    prediction = get_prediction(canvas_result.image_data)
    st.text("Prediction : {}".format(prediction))


  


