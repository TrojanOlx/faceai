from flask_restful import Resource, request, reqparse
from common.util import make_result
from .compute.face_contrast import FaceRecognition
from .compute.face_rtmp import FaceRtmpOut

# is face contrast api

face_recognition=FaceRecognition()

face_rtmpOut=FaceRtmpOut()


class FaceContrast(Resource):
    def get(self):
        data = {
            "import": {
                "image1": "F:/image1.jpg",
                "image2": "F:/image2.jpg"
            },
            "out": {
                "code": 1200,
                "data": {
                    "image1": {
                        "url": "F:/image1.jpg"
                    },
                    "image2": {
                        "url": "F:/image2.jpg"
                    },
                    "similarity": 0.5
                },
                "msg": "success"
            }
        }
        return data,200

    def post(self):
        # Get the data from the request
        json_data = request.get_json(force=True)
        image1 = json_data['image1']
        image2 = json_data['image2']

        
        similarity,detectorlist=face_recognition.score(image1,image2)
        data = {
            "similarity": similarity,
            "image1": detectorlist[0],
            "image2": detectorlist[1]
        }
        return make_result(data=data)



class FaceRetrieve(Resource):
    def get(self):
        return make_result(data={})
    
    def post(self):
        json_data=request.get_json(force=True)
        image=json_data['image']
        images=list(json_data['images'])

        face,faces=face_recognition.score_list(image,images)
        data={
            "image":face,
            "images":faces
        }
        return make_result(data=data)



class FaceRtmp(Resource):
    def get(self):
        return ""

    def post(self):
        json_data=request.get_json(force=True)
        rtmpurl=json_data['rtmp']
        url = face_rtmpOut.initRtmp(rtmpurl)
        data={
            "url":url
        }
        return make_result(data=data)


