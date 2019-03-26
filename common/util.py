from flask import jsonify
from common.code import Code


def make_result(data=None, code=Code.SUCCESS):
    return jsonify({"code": code, "data": data, "msg": Code.msg[code]})