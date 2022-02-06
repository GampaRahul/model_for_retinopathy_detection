import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import cv2
from skimage import exposure
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
import numpy as np

#model = load_model('C:\\Users\\gampa\\PycharmProjects\\MajorProject\\venv\\model.h5')


def intensity_slicing(grayimage, layers=7):
    grayimage = cv2.addWeighted(grayimage, 4, cv2.GaussianBlur(grayimage, (0, 0), 15), -4, 128)
    grayimage = exposure.equalize_hist(grayimage)
    return grayimage


def predict(img):
    model = load_model('C:\\Users\\gampa\\PycharmProjects\\MajorProject\\venv\\model.h5')
    BATCH_SIZE = 32
    print("In Predicting")
    image = img
    test = pd.DataFrame([[image]], columns=['id_code'])
    test_dir = "C:\\Users\\gampa\Desktop\\aptos2019-blindness-detection\\test_images"
    test_data_gen = ImageDataGenerator(preprocessing_function=intensity_slicing)
    test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test,
        directory=test_dir,
        x_col="id_code",
        target_size=(300, 300),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode=None
    )

    predicted = model.predict_generator(test_generator, steps=len(test_generator.filenames) / BATCH_SIZE)
    test['diagnosis'] = np.argmax(predicted, axis=1)
    test['id_code'] = test['id_code'].apply(lambda x: x[:-4])
    severity = test['diagnosis'].iloc[0]
    print(severity)
    return severity