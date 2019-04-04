import dlib
import cv2.cv2 as cv2
import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf


detector = dlib.get_frontal_face_detector()


def get_detector(img):
    faces = detector(img, 1)
    return faces


def read_img(img_src):
    img_srcs = glob.glob(img_src)
    if len(img_srcs) <= 0:
        return None, "src null"
    return cv2.imread(img_srcs[0]), None


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


emotion_offsets = (20, 40)
gender_offsets = (10, 10)
detection_model_path = './apps/v1/compute/model/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './apps/v1/compute/model/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = './apps/v1/compute/model/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')


# emotion_classifier = load_model(emotion_model_path, compile=False)
# gender_classifier = load_model(gender_model_path, compile=False)


class drawImg:
    def __init__(self):
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.gender_classifier = load_model(gender_model_path, compile=False)
        self.graph = tf.get_default_graph()
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.gender_target_size = self.gender_classifier.input_shape[1:3]
    def load_image(self, image_path, grayscale=False, target_size=None):
        color_mode = 'grayscale'
        if grayscale == False:
            color_mode = 'rgb'
        else:
            grayscale = False
        pil_image = image.load_img(
            image_path, grayscale, color_mode, target_size)
        return image.img_to_array(pil_image)

    def Draw(self, image):

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 人脸检测
        faces = self.GetFace(gray_image)

        for face_coordinates in faces:
            x, y, width, height = face_coordinates

            try:
                cv2.rectangle(image, (x,y ), (x+width, y+height), (255, 0, 0), 2)


                # 性别检测
                x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]
                gender_text = self.GetSex(rgb_face)

                # 情绪检测
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                emotion_face = gray_image[y1:y2, x1:x2]
                emotion_text = self.GetEmotion(emotion_face)

                cv2.putText(image, "G:{},E:{}".format(gender_text, emotion_text),
                            (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            except:
                continue
        return image

    # 人脸检测
    def GetFace(self, img):
        faces = self.face_detection.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(
            60, 60), maxSize=(300, 300), flags=cv2.CASCADE_SCALE_IMAGE)
        
        return faces

    # 性别检测
    def GetSex(self, img):
        rgb_face = cv2.resize(img, (self.gender_target_size))
        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        with self.graph.as_default():
            gender_prediction = self.gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
            return gender_text
        return ""

    # 情绪检测
    def GetEmotion(self, img):
        gray_face = cv2.resize(img, (self.emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        with self.graph.as_default():
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            return emotion_text
        return ""

