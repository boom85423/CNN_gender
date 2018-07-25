from keras.models import load_model
import cv2
import time
from CNN_gender_train import detect_head
from keras.models import Sequential

def predict_gender(image_path):
    image = detect_head(image_path)
    image = image.reshape(1,1,666,666)
    gender = model.predict_classes(image)
    if gender == 0:
        return 'female'
    else:
        return 'male'

def revise_model(image_path='selfie.jpg', image_label):
    image = detect_head(image_path)
    image = image.reshape(1,1,666,666)
    image_label = np_utils.to_categorical(image_label, num_classes=2).reshape(1,2)
    # Force model to learn the image.
    model.fit(image, image_label, nb_epoch=3)

if __name__ == '__main__':
    model = load_model('cnn_gender_model.h5')
    # selfie
    camera = cv2.VideoCapture(0)
    time.sleep(1)
    selfie = camera.read()[1]
    camera.release()
    cv2.imwrite('selfie.jpg', selfie)
    print(predict_gender('selfie.jpg'))
    # revise_model(image_label=1)
