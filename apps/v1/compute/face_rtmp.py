import random
import string
from multiprocessing import Process
import cv2.cv2 as cv2
import subprocess as sp
import threading

# push rtmp url

rtmpUrl='rtmp://127.0.0.1:1935/live/'
detection_path = "./apps/v1/compute/model/detection_models/haarcascade_frontalface_default.xml"

# 读取
face_detection =cv2.CascadeClassifier(detection_path)


rtmpdist={}


def cs(index,url):
    print("{}{}".format(index,url))


def Tortmp(index,url):
        print(url+"\n")
        camera=cv2.VideoCapture(index)
        # 视频属性
        size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        sizeStr = str(size[0]) + 'x' + str(size[1])
        fps = camera.get(cv2.CAP_PROP_FPS)  # 30p/self
        fps = int(fps)
        hz = int(1000.0 / fps)
        # 管道输出 ffmpeg推送rtmp 重点 ： 通过管道 共享数据的方式
        command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', sizeStr,
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv', 
            url]

        pipe = sp.Popen(command, stdin=sp.PIPE)

        count=0
        while True:
            ret, frame = camera.read() # 逐帧采集视频流
            if not ret:
                count+=1
                if count>60:break
                continue
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 人脸检测
            faces=face_detection.detectMultiScale(gray_image,1.3,5)
            for face_coordinates in faces:
                x, y, width, height = face_coordinates
                cv2.rectangle(frame, (x ,y ), (x + width , y + height), (255,0,0), 2)
            
            pipe.stdin.write(frame.tostring())  # 存入管道

class FaceRtmpOut:
    def initRtmp(self,rtmpurl):
        uid="".join(random.sample(string.ascii_letters+string.digits,6))
        url=rtmpUrl+uid
        p=Process(target=Tortmp,args=(rtmpurl,url,))
        p.start()
        rtmpdist[rtmpurl]=p
        return url
    def removeRtmp(self,rtmpUrl):
        rtmpdist[rtmpUrl].terminate()
        del rtmpdist[rtmpUrl]
