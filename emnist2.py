import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image #Отрисовка изображений

st.set_page_config(layout="wide")
st.header("Распознавание букв с помощью нейронной сети")

with st.expander('Рассмотрим как выглядят буквы и цифры из базы'):
    image = Image.open('numbersletters.jpg')
    st.image(image)

testovic = np.load('test.npz')
X_test_ = testovic['x']
x_test_2D = X_test_ / 255 # делим на 255, чтобы диапазон был от 0 до 1

model_2d = load_model('/app/emnist1/model_emnist.h5')

dictant1 = {0:'0' , 1:'1' , 2:'2' , 3:'3' , 4:'4' , 5:'5' , 6:'6' , 7:'7' , 8:'8' , 9:'9' , 10:'A' , 11:'B' ,
            12:'C' , 13:'D' , 14:'E' , 15:'F' , 16:'G' , 17:'H' , 18:'I' , 19:'i' , 20:'k' , 21:'l' , 22:'M' ,
            23:'N' , 24:'O' , 25:'P' , 26:'Q' , 27:'R' , 28:'S' , 29:'T' , 30:'U' , 31:'V' , 32:'W' , 33:'X' ,
            34:'Y' , 35:'Z' , 36:'a' , 37:'b' , 38:'d' , 39:'e' , 40:'f' , 41:'g' , 42:'h' , 43:'n' , 44:'q' ,
            45:'r' , 46:'t'}

def img_preprocess(img):
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    # st.write(type(img_array))

    # Check the shape of img_array:
    # Should output shape: (height, width, channels)
    # st.write(img_array.shape)

    # make square shape
    img_height, img_width = img_array.shape[0], img_array.shape[1]
    img_center = int(img_width / 2)
    left_border = int(img_center - img_height / 2)
    right_border = int(img_center + img_height / 2)
    img_array1 = img_array[:, left_border:right_border, :]

    # Check the shape of img_array:
    # st.write(img_array1.shape)

    # convert n save
    im = Image.fromarray(img_array1)
    im.save('your_file_image.png')
    # image11 = Image.open('your_file_image.png')
    image11 = Image.open('your_file_image.png')
    img11 = image11.resize((28, 28), Image.ANTIALIAS)

    # convert image to one channel & Numpy array
    img12 = img11.convert("L")
    imgData = np.asarray(img12)

    # Calculate THRESHOLD_VALUE
    # assume dark digit & white sheet
    step_lobe = .4
    mid_img_color = np.sum(imgData) / imgData.size
    min_img_color = imgData.min()

    THRESHOLD_VALUE = int(mid_img_color - (mid_img_color - min_img_color) * step_lobe)

    print(mid_img_color)
    print(min_img_color)
    print(THRESHOLD_VALUE)

    thresholdedData = (imgData < THRESHOLD_VALUE) * 1.0
    imgData1 = np.expand_dims(thresholdedData, axis=0)
    return imgData1

col21 , col22 = st.columns(2)
with col21:
    with st.container():
        st.subheader("Распознавание букв из тестовой выборки")
        y_test_2D_pred = model_2d.predict(x_test_2D)
        y_test_2D_pred


with col22:
    with st.container():
        st.subheader("Распознавание фото")
        img_file_buffer = st.camera_input("Фото")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)

            # To convert PIL Image to numpy array:
            img_array = np.array(img)

            mnist_like = img_preprocess(img_array)

            #st.write(imgData1)

            y_predict1 = model_2d.predict(mnist_like)
            y_maxarg = np.argmax(y_predict1, axis=1)

            st.write(y_predict1)
            st.write('Нейронная сеть считает, что это ', dictant1[int(y_maxarg)] )
            #st.subheader(dictant1[y_maxarg])




