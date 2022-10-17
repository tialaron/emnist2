import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image,  ImageEnhance, ImageFilter #Отрисовка изображений
from img_preproc import img_preprocess
import os

model_2d = load_model('/app/emnist2/model_emnist.h5')
file_path = '/app/emnist2/your_file_image.png'


#st.set_page_config(layout="wide")
st.markdown('''<h1 style='text-align: center; color: #000000;'
            >Распознавание рукописных букв нейронной сетью</h1>''', 
            unsafe_allow_html=True)

img_start = Image.open('/app/emnist2/pictures/start_picture.png') #
st.image(img_start, use_column_width='auto') #width=450

st.write("""
Лабораторная работа *"Распознавание рукописных букв нейронной сетью "* позволяет продемонстрировать 
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
with st.expander('Области возможного применения:'):
            st.write('1.Чтение заявлений, служебных записок и других документов с последующим переносом их в печатный текст. Будет актуально для студентов, обучающихся по административному направлению.')
            st.write('2.Чтение и распознавание исторических рукописных документов. Будет актуально для студентов исторических факультетов.')
            st.write('3.Автоматическое заполнение баз данных при рассмотрении письменных показаний свидетелей. Пригодится студентам юридических факультетов.')
            st.write('4.Будет интересна студентам других специальностей.')
testovic = np.load('test.npz')
X_test_ = testovic['x']
x_test_2D = X_test_ / 255 # делим на 255, чтобы диапазон был от 0 до 1

dictant1 = {0:'0' , 1:'1' , 2:'2' , 3:'3' , 4:'4' , 5:'5' , 6:'6' , 7:'7' , 8:'8' , 9:'9' , 10:'A' , 11:'B' ,
            12:'C' , 13:'D' , 14:'E' , 15:'F' , 16:'G' , 17:'H' , 18:'I' , 19:'i' , 20:'k' , 21:'l' , 22:'M' ,
            23:'N' , 24:'O' , 25:'P' , 26:'Q' , 27:'R' , 28:'S' , 29:'T' , 30:'U' , 31:'V' , 32:'W' , 33:'X' ,
            34:'Y' , 35:'Z' , 36:'a' , 37:'b' , 38:'d' , 39:'e' , 40:'f' , 41:'g' , 42:'h' , 43:'n' , 44:'q' ,
            45:'r' , 46:'t'}

st.write('Нейронная сеть, представленная здесь, обучена с помощью алгоритма "Обучение с учителем". В качестве входных данных используется стандартная база данных изображений рукописных букв и цифр [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset).')
st.write('Выходными данными нейронной сети являются вероятностные значения, полученные на нейронах. При этом каждый нейрон воспроизводит вероятность того класса за который отвечает.')
st.write('Данная модель является категориальной. Гибкостью она не обладает. Обучение проводится не в процессе работы сети, а в отдельной среде с последующим копированием на данную платформу.')
st.subheader('Пункт 1.')
st.write('Вам предоставляется на выбор два варианта выполнения работы.'
         ' Вы можете самостоятельно сделать фотоснимок буквы/цифры (левая колонка), либо воспользоваться готовыми изображенями (правая колонка).')
choice1 = st.radio("Видео или готовые изображения?",('Видео', 'Изображения'))

col1,col2 = st.columns(2)
with col1:
            st.write('Одной рукой поднесите карточку с изображением к видеокамере так, чтобы она занимала большую часть экрана,'
                     ' а другой рукой возьмите мышь и щёлкните на кнопку под изображением')
            img_file_buffer = st.camera_input("Take picture")
            
            #img = Image.open(img_file_buffer)
            #img_array = np.array(img)
            #img_height, img_width = img_array.shape[0], img_array.shape[1]
            #img_center = int(img_width / 2)
            #left_border = int(img_center - img_height / 2)
            #right_border = int(img_center + img_height / 2)
            #img_array1 = img_array[:, left_border:right_border, :]
            #im = Image.fromarray(img_array1)

with col2:
            st.write('Вы можете выбрать любое изображение из предложенных ниже.')
            option1 = st.selectbox('Какое Вы выбираете?',('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
                                                                'I','_i','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                                                                '_a','_b','_d','_e','_f','_g','_h','_n','_q','_r','_t'))
            pict_path = '/app/emnist2/test_pict/foto'+option1+'.png'
            img_pict = Image.open(pict_path)
            st.image(pict_path)

if choice1 == 'Видео' and img_file_buffer is not None: 
            img = Image.open(img_file_buffer)
            img_array = np.array(img)
            img_height, img_width = img_array.shape[0], img_array.shape[1]
            img_center = int(img_width / 2)
            left_border = int(img_center - img_height / 2)
            right_border = int(img_center + img_height / 2)
            img_array1 = img_array[:, left_border:right_border, :]
            im = Image.fromarray(img_array1)
            im.save(file_path)
if choice1 == 'Изображения':
            img_pict.save(file_path)
            
            
st.write('Пункт 2.')
st.write('Зарисуйте полученное изображение чёрно-белого изображения из окошка в бланк отчёта. '
         'Необходимо на рисунке отобразить возникшие недочёты изображения, например, пропуски. Чтобы'
         ' не зарисовывать всё чёрное пространство, рекомендуется изобразить ручкой на белом фоне '
         'листа бланка отчёта.')

st.write('Пункт 3.')
st.write('Нажмите на кнопку распознавания, запишите результат.')
isbutton1 = st.button('Распознать')
col3, col4 = st.columns(2)
with col3:      
              st.write('Вот что увидела нейронная сеть.')
              if isbutton1:
                          image11 = Image.open(file_path)
                          st.image(file_path) 
                          img11 = image11.resize((28, 28), Image.LANCZOS)   
                          img11.save(file_path)                        
                          imgData1 = np.expand_dims(np.asarray(img11.convert("L")), axis=0)

with col4:
              st.write('Она распознала это как...')
              if isbutton1:
                          y_predict1 = model_2d.predict(imgData1) 
                          y_maxarg = np.argmax(y_predict1, axis=1)
                          st.subheader(dictant1[int(y_maxarg)])
         


st.write('Пункт 4.')
st.write('Включите коррекцию яркости. Посмотрите, улучшило ли это изображение негатива.'
         ' Зарисуйте результат, как указано выше.')
col5,col6 = st.columns(2)
with col5:
         value_sli = st.slider('Коррекция яркости', 0.0, 100.0, 50.0)
with col6:
         st.write('Яркость',value_sli)
         image111 = Image.open(file_path)
         enhancer = ImageEnhance.Brightness(image111)
         factor = 2*value_sli / 100 #фактор изменения
         im_output = enhancer.enhance(factor)
         im_output.save(file_path)
         st.image(file_path)   

st.write('Пункт 5.')
st.write('Нажмите на кнопку распознавания, запишите результат.')
isbutton2 = st.button('Распознать еще картинку')
col7,col8 = st.columns(2)
with col7:
             if isbutton2:
                   st.image(file_path)
with col8:
             if isbutton2:
                   image112 = Image.open(file_path)
                   img111 = image112.resize((28, 28), Image.LANCZOS)  
                   img121 = img111.convert("L")
                   imgData = np.asarray(img121)
                   step_lobe = value_sli / 100
                   mid_img_color = np.sum(imgData) / imgData.size
                   min_img_color = imgData.min()
                   THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
                   thresholdedData = (imgData < THRESHOLD_VALUE) * 1.0
                   imgData1 = np.expand_dims(thresholdedData, axis=0)
                   y_predict1 = model_2d.predict(imgData1)
                   y_maxarg = np.argmax(y_predict1, axis=1)
                   st.subheader(dictant1[int(y_maxarg)])  

st.write('Пункт 6.')
st.write('Скорректируйте изображение с помощью фильтра Гаусса. Нажмите на кнопку распознавания, запишите результат.')
col9,col10 = st.columns(2)
with col9:
            value_gaus = st.slider('Фильтр Гаусса', 0, 10, 0)
with col10:
            st.write('Фильтр Гаусса',value_gaus)
            image222 = Image.open(file_path)
            im2 = image222.filter(ImageFilter.GaussianBlur(radius = value_gaus))
            im2.save(file_path)
            st.image(file_path)
            
st.write('Пункт 7.')
st.write('Попробуем теперь еще раз распознать картинку.')
isbutton3 = st.button('Распознать картнку еще раз')
col11,col12 = st.columns(2)
with col11:
            if isbutton3:
                   st.image(file_path)
with col12:
            if isbutton3:
                   image333 = Image.open(file_path)
                   img333 = image333.resize((28, 28), Image.LANCZOS) 
                   img334 = img333.convert("L")
                   imgData4 = np.asarray(img334) 
                   step_lobe = value_sli / 100
                   mid_img_color = np.sum(imgData4) / imgData4.size
                   min_img_color = imgData4.min()
                   THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
                   thresholdedData = (imgData4 < THRESHOLD_VALUE) * 1.0
                   imgData5 = np.expand_dims(thresholdedData, axis=0)
                   y_predict2 = model_2d.predict(imgData5)
                   y_maxarg2 = np.argmax(y_predict2, axis=1)
                   st.subheader(dictant1[int(y_maxarg2)]) 
                    
st.write('Пункт 8.')
st.write('Сделайте выводы, какие именно фильтры и как влияют на результат эксперимента')
st.write('Пункт 9.')
st.write('Посмотрим как "видит" картинку нейронная сеть')
col13,col14 = st.columns(2)
with col13:
         value_thres = st.slider('Порог отсечки', 0, 100, 50)
with col14:
         st.write('Порог отсечки',value_thres)
         image444 = Image.open(file_path)
         i2 = image444.convert("L")
         i3 = np.asarray(i2)
         step_lobe = value_thres / 100
         mid_img_color = np.sum(i3) / i3.size
         min_img_color = i3.min()
         THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
         thresholdedData = (i3 < THRESHOLD_VALUE) * 255.0
         imm1 = Image.fromarray(thresholdedData)
         imm1 = imm1.convert("L")
         imm1.save(file_path)
         st.write(imm1) 
         st.image(file_path)
       
st.write('Пункт 10. ')
st.write('Ответьте на вопросы. ')
st.write('1. Распознала ли нейронная сеть цифру с первого раза? ')
st.write('2. Как повлияло изменение яркости на результат? (Улучшило/Ухудшило/Никак не повлияло) ')
st.write('3. Как повлияло применение фильтра Гаусса на результат? (Улучшило/Ухудшило/Никак не повлияло) ')
st.write('4. Попробуйте провести несколько экспериментов с разными цифрами меняя только значения фильтра Гаусса.'
         ' На Ваш взгляд стоит ли его использовать при корректировке изображения?')
st.write('5. Посмотрите на черно-белое изображение где показано как "видит" цифру нейронная сеть.'
         ' Сравните с изображениями обучающей и тестовой выборки, которые есть на картинке в начале работы.'
         ' Насколько Ваша картинка похожа на эти изображения?')
st.write('')
st.write('Дополнительная литература:')
st.write('1) Марк Лутц "Изучаем Python". https://codernet.ru/books/python/izuchaem_python_5-e_izd_tom_1_mark_lutc/ ')
st.write('2) Документация библиотеки Keras. https://keras.io/ ')
st.write('3) Документация библиотеки Streamlit. https://docs.streamlit.io/ ')
st.write('')
st.write('Пожелания и замечания')            
st.write('https://docs.google.com/spreadsheets/d/1GWCusE2WyCN8R7iqGaOEA9gwT6UanryXRQPo4tdID0M/edit?usp=sharing')



