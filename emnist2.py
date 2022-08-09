import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image,  ImageEnhance, ImageFilter #Отрисовка изображений
import img_preproc

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




