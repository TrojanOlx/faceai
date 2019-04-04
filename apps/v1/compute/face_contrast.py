import dlib
import cv2.cv2 as cv2
import glob
import numpy as np


predictor_path = "./apps/v1/compute/model/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./apps/v1/compute/model/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()

sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


class FaceRecognition:

    def face_detection_one(self, img_path):
        dist = []
        detectors = []
        img = cv2.imread(img_path)
        # 转换rgb顺序的颜色。
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        # 检测人脸
        faces = detector(img, 1)
        if len(faces):
            for index, face in enumerate(faces):
                # # 提取68个特征点
                shape = sp(img2, face)
                # 计算人脸的128维的向量
                face_descriptor = facerec.compute_face_descriptor(img2, shape)
                dist.append(list(face_descriptor))
                detectors.append(faces)
        else:
            pass
        return dist, detectors

    def face_detection(self, url_img_1, url_img_2):
        img_path_list = [url_img_1, url_img_2]
        dist = []
        detectors = []
        for img_path in img_path_list:
            img = cv2.imread(img_path)
            # 转换rgb顺序的颜色。
            b, g, r = cv2.split(img)
            img2 = cv2.merge([r, g, b])
            # 检测人脸
            faces = detector(img, 1)
            if len(faces):
                for index, face in enumerate(faces):
                    # # 提取68个特征点
                    shape = sp(img2, face)
                    # 计算人脸的128维的向量
                    face_descriptor = facerec.compute_face_descriptor(
                        img2, shape)
                    dist.append(list(face_descriptor))
                    detectors.append(faces)
            else:
                pass
        return dist, detectors

    def face_detection_list(self, img_path_list):
        dist = []
        detectors = []
        for img_path in img_path_list:
            img = cv2.imread(img_path)
            # 转换rgb顺序的颜色。
            b, g, r = cv2.split(img)
            img2 = cv2.merge([r, g, b])
            # 检测人脸
            faces = detector(img, 1)
            if len(faces):
                for index, face in enumerate(faces):
                    # # 提取68个特征点
                    shape = sp(img2, face)
                    # 计算人脸的128维的向量
                    face_descriptor = facerec.compute_face_descriptor(
                        img2, shape)
                    dist.append(list(face_descriptor))
                    detectors.append(faces)
            else:
                pass
        return dist, detectors

    # 欧式距离
    def dist_o(self, dist_1, dist_2):
        dis = np.sqrt(sum((np.array(dist_1)-np.array(dist_2))**2))
        return dis

    def score(self, url_img_1, url_img_2):
        try:
            url_img_1 = glob.glob(url_img_1)[0]
            url_img_2 = glob.glob(url_img_2)[0]
            data, detectors = self.face_detection(url_img_1, url_img_2)
            goal = self.dist_o(data[0], data[1])
        except Exception as e:
            print(str(e))

        detectorlist = []
        for i, d in enumerate(detectors):
            d = d[0]
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            detectorlist.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h
            })
        # 判断结果，如果goal小于0.6的话是同一个人，否则不是。我所用的是欧式距离判别
        return 1-goal, detectorlist

    def score_list(self, url_img, url_images):
        data, detector = self.face_detection_one(url_img)
        datas, detectors = self.face_detection_list(url_images)

        coordinate = {}

        for i, d in enumerate(detector):
            d = d[0]
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            coordinate = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h
            }
        detectorlist = []
        for i, d in enumerate(detectors):
            d = d[0]
            goal = self.dist_o(data[0], datas[i])
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            detectorlist.append(
                {
                    "similarity": 1-goal,
                    "coordinate": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "w": w,
                        "h": h
                    }
                })
        return coordinate,detectorlist

    def score_one(self,url_img):
        data, detector = self.face_detection_one(url_img)
        goal=0
        detectorlist = []
        for i, d in enumerate(detector):
            d = d[0]
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            detectorlist.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h
            })
        if len(data)>=2:
            goal = self.dist_o(data[0], data[1])

        return 1-goal, detectorlist
            