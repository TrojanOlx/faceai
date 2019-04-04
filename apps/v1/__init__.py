from flask import Blueprint
from flask_restful import Api
from apps.v1.face import FaceContrast,FaceRetrieve,FaceIDCard,FaceRtmp

def register_views(app):
    api = Api(app)
    api.add_resource(FaceContrast,'/faceContrast',endpoint="faceContrast")
    api.add_resource(FaceRetrieve,'/faceRetrieve',endpoint='faceRetrieve')
    api.add_resource(FaceIDCard,'/faceIDCard',endpoint='faceIDCard')
    api.add_resource(FaceRtmp,'/facertmp',endpoint='facertmp')
def create_blueprint_v1():
    """
    注册蓝图->v1版本
    """
    bp_v1 = Blueprint('v1', __name__)
    register_views(bp_v1)
    return bp_v1