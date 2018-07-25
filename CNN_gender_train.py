import cv2
from skimage import color
import os
import glob
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import multiprocessing as mp
import time

def detect_head(image_path):
    # Use opencv feature extractor to detect face.
    image = cv2.imread(image_path)
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    image = image[faces[0][1]:faces[0][1]+faces[0][2], faces[0][0]:faces[0][0]+faces[0][3]]
    image = cv2.resize(image, (666,666))
    image = color.rgb2grey(image)
    image = image.reshape(1,666,666)
    return image

if __name__ == '__main__':
    pool = mp.Pool()
    # female datasets
    female_path = 'female/'
    detect_female_job = [pool.apply_async(detect_head, args=(i,)) for i in glob.glob(os.path.join(female_path, '*.JPG'))]  
    female = []
    for i in detect_female_job:
        try:
            female.append(i.get()[0]) # lazy function
        except: # fail to detect
            pass
    # male datasets
    male_path = 'male/'
    detect_male_job = [pool.apply_async(detect_head, args=(i,)) for i in glob.glob(os.path.join(male_path, '*.JPG'))]  
    male = []
    for i in detect_male_job:
        try:
            male.append(i.get()[0])
        except: 
            pass
    # preprocessing
    heads = female + male
    heads = np.asarray(heads)
    x = heads.reshape(len(heads),1,666,666)
    gender_female = np.repeat([0],len(female))
    gender_male = np.repeat([1],len(male))
    gender = np.append(gender_female, gender_male)
    y = np_utils.to_categorical(gender, num_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=int(len(x)*0.2))
    # model training
    model = Sequential([
        # convolution1
        Dropout(rate=0.2),
        Convolution2D(nb_filter=36, nb_row=6, nb_col=6, border_mode='same', input_shape=(1,666,666)),
        Activation('relu'),
        MaxPooling2D(pool_size=(6,6), strides=(6,6), border_mode='same'),
        # convolution2
        Convolution2D(nb_filter=72, nb_row=6, nb_col=6, border_mode='same', input_shape=(1,666,666)),
        Activation('relu'),
        MaxPooling2D(pool_size=(6,6), strides=(6,6), border_mode='same'),
        # fully1
        Flatten(),
        Dense(1024),
        Activation('relu'),
        # fuly2
        Dense(2),
        Activation('softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    t1 = time.time()
    while True:
        model.fit(x_train, y_train, nb_epoch=3, batch_size=10)
        loss, accuracy = model.evaluate(x_test, y_test)
        print('Prediction accuracy : {}'.format(accuracy))
        if accuracy >= 0.9: # Keep training until accuracy reach 90% confidence level.
            model.save('CNN_gender_model.h5')
            break
    print('Total time : {}s'.format(time.time()-t1))
