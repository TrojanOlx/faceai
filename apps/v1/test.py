import cv2.cv2 as cv2
import gevent
from flask import Flask, Response, render_template
from flask_restful import Resource, reqparse, request

import dlib
from common.util import make_result

from gevent import monkey; monkey.patch_all()

parser = reqparse.RequestParser()
detector=dlib.get_frontal_face_detector()

class Hello(Resource):
    def get(self):
        return make_result(data={"hello":"Word"})

    def post(self):
        json_data = request.get_json(force=True)
        data=json_data["data"]["hello"]
        return make_result(data=data)
    


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0) 
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()

        detected=detector(image,1)
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # cv2.putText(image, 
        #             "S:{} E:{} A:{}".format(1,2,3), 
        #             (100, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255,0,0), 1, cv2.LINE_AA) \d{3}[.]\d{3}[.]\d{3}
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

#@app.route('/video')  # 主页
def video():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    #return Response(gevent.spawn(gen(VideoCamera())),mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')   

#@app.route('/video_gevent')
def video_gevent():
    gevent.spawn(video_feed())
