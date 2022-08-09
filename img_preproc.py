import numpy as np
from PIL import Image

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
