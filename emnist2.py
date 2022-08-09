import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image,  ImageEnhance, ImageFilter #Отрисовка изображений
from img_preproc import img_preprocess

model_2d = load_model('/app/emnist2/model_emnist.h5')
file_path = '/app/emnist2/your_file_image.png'

#st.set_page_config(layout="wide")
st.markdown('''<h1 style='text-align: center; color: #F64A46;'
            >Распознавание рукописных букв искусственной нейронной сетью (ИНС)</h1>''', 
            unsafe_allow_html=True)

img_start = Image.open('/app/emnist2/pictures/start_picture.png') #
st.image(img_start, use_column_width='auto') #width=450

st.write("""
Лабораторная работа *"Распознавание рукописных букв искусственной нейронной сетью (ИНС)"* позволяет продемонстрировать 
функционирование реальной нейронной сети, обученной распознавать рукописные буквы. Для обучения использовалась база EMNIST, содержащая буквы латинского алфавита.
[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
""")
st.write('Вот так выглядят буквы и цифры из базы')
image = Image.open('/app/emnist2/pictures/numbersletters.jpg')
st.image(image)

with st.expander("Общая схема"):
  img_pipeline_mnist = Image.open('/app/emnist2/pictures/pipeline_for_MNIST_4.png') 
  st.image(img_pipeline_mnist, use_column_width='auto', caption='Общая схема лабораторной работы') #width=450 
  st.markdown(
    '''
    \n**Этапы:**
    \n1. База данных EMNIST, содержащая образцы рукописных букв:
    \nСодержит 800000 картинок размером 28x28 пикселей с изображением рукописных букв. [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
    \n2. Библиотека слоев:
    \nСодержит набор слоев, используемых нейронной сетью.  [tensorflow](https://www.tensorflow.org/).
    \n3. Настройка модели:
    \nУстанавливается тип и количество слоев, а также количество нейронов в них.
    \n4. Обучение модели:
    \nВо время этого процесса нейросеть просматривает картинки и сопоставляет их с метками.
    \n5. Проверка точности:
    \nНа этом этапе программист проверяет работу сети с помощью тестовых изображений.
    \n6. Функция обработки изображения:
    \nПреобразует изображение в массив чисел, который понимает нейронная сеть.
    \n7. Загрузка изображения:
    \nНа выбор студенту предлагается два варианта ввода. Первый - ввести данные с помощью камеры, второй - воспользоваться заранее отобранными изображениями.
    \n8. Проверка:
    \nДалее нужно проверить на сколько правильно нейронная сеть распознала букву.
    \n9. Корректировка:
    \nУправляя яркостью, размытостью изображения(фильтр Гаусса) и контрастностью (порог отсечки) нужно добиться распознавания изображения нейронной сетью. 
    \n10. Приложение Streamlit:
    \nОтображение результатов работы нейронной сети.
    ''')         

with st.expander("Краткое описание искусственной нейронной сети и ее работы."):
            st.write('Искусственная нейронная сеть - это математическая модель настоящей нейронной сети,'  
                     'то есть мозга. На практике, это обучаемый под требуемую задачу инструмент.' 
                     'Искусственная нейронная сеть представляет собой набор матриц, с которыми работают по законам линейной алгебры. '
                     'Тем не менее, проще представить её как набор слоёв нейронов, связанных между собой '
                     'за счёт входных и выходных связей.')
            st.image('/app/emnist2/pictures/fully_connected_NN.png', caption='Пример построения нейронной сети')
            st.write('Различают внешние слои - входной и выходной, и внутренние, находящиеся между ними. '
                     'У каждого отдельного нейрона, например, перцептрона, может быть несколько входных связей, у каждой из связей - свой множитель усиления' 
                     '(ослабления) влияния связи - весовой коэффициент, или вес. На выходе нейрона действует функция активации, за счёт нелинейностей ' 
                     'функций активации и подбора параметров-весов на входе нейрона, нейронная сеть и может обучаться.')
            st.image('/app/emnist2/pictures/activation_functions.png',caption='Набор активационных функций')

st.markdown('''<h1 style='text-align: center; color: black;'
            >Задача лабораторной работы.</h1>''', 
            unsafe_allow_html=True)
st.write('  Возможность распознавать образы является одним из признаков интеллекта. Для компьютеров это уже не является сложной задачей.'
         'В данной работе Вам предстоит проверить насколько хорошо обучена нейронная сеть распознавать рукописные буквы. '
         'Это может пригодиться для создания программ прочтения и перевода в печатный текст рукописей или рецептов врача.')
            
testovic = np.load('test.npz')
X_test_ = testovic['x']
x_test_2D = X_test_ / 255 # делим на 255, чтобы диапазон был от 0 до 1

dictant1 = {0:'0' , 1:'1' , 2:'2' , 3:'3' , 4:'4' , 5:'5' , 6:'6' , 7:'7' , 8:'8' , 9:'9' , 10:'A' , 11:'B' ,
            12:'C' , 13:'D' , 14:'E' , 15:'F' , 16:'G' , 17:'H' , 18:'I' , 19:'i' , 20:'k' , 21:'l' , 22:'M' ,
            23:'N' , 24:'O' , 25:'P' , 26:'Q' , 27:'R' , 28:'S' , 29:'T' , 30:'U' , 31:'V' , 32:'W' , 33:'X' ,
            34:'Y' , 35:'Z' , 36:'a' , 37:'b' , 38:'d' , 39:'e' , 40:'f' , 41:'g' , 42:'h' , 43:'n' , 44:'q' ,
            45:'r' , 46:'t'}


st.subheader('Пункт 1.')
st.write('Вам предоставляется на выбор два варианта выполнения работы.'
         ' Вы можете самостоятельно сделать фотоснимок буквы/цифры (левая колонка), либо воспользоваться готовыми изображенями (правая колонка).')
choice1 = st.radio("Видео или готовые изображения?",('Видео', 'Изображения'))

col1,col2 = st.columns(2)
with col1:
            st.write('Одной рукой поднесите цифру к видеокамере так, чтобы она занимала большую часть экрана,'
                     ' а другой рукой возьмите мышь и щёлкните на кнопку под изображением')
            img_file_buffer = st.camera_input("Take picture")
            #if choice1 == 'Видео':
            img = Image.open(img_file_buffer)
            img_array = np.array(img)
            img_height, img_width = img_array.shape[0], img_array.shape[1]
            img_center = int(img_width / 2)
            left_border = int(img_center - img_height / 2)
            right_border = int(img_center + img_height / 2)
            img_array1 = img_array[:, left_border:right_border, :]
            im = Image.fromarray(img_array1)
            if choice1 == 'Видео' and img_file_buffer is not None: 
                        im.save(file_path)
with col2:
            st.write('Вы можете выбрать любое изображение из предложенных ниже.')
            option1 = st.selectbox('Какое Вы выбираете?',('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
                                                                'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                                                                'a','b','d','e','f','g','h','n','q','r','t'))
            pict_path = '/app/emnist2/test_pict/foto'+option1+'.png'
            img = Image.open(pict_path)
            st.image(pict_path)
            
            if choice1 == 'Изображения' and img_file_buffer == None:
                        #pict_path = '/app/emnist2/test_pict/foto'+option1+'.png'
                        #img = Image.open(pict_path)
                        #st.image(pict_path)
                        
                        img.save(file_path)
            

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




