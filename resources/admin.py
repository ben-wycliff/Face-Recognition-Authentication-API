# from flask import Blueprint, Response, request, jsonify
# # from database.models import Admin
# import json
# import pickle
# import face_recognition
# import glob
# import numpy as np
#
# image = face_recognition.load_image_file('./image.jpg')
# model = pickle.load(open('./classifier.sav', 'rb'))
#
# admin = Blueprint('admin', __name__)
#
# # routes
#
#
# @admin.route('/detect', methods=["POST"])
# def face_detect():
#     face_location = face_recognition.face_locations(image, number_of_times_to_upsample=2)
#     face_encodings = face_recognition.face_encodings(image, known_face_locations=face_location)[0]
#     encodings_array = np.asarray(face_encodings)
#     prediction = model.predict(encodings_array)
#     return jsonify(prediction), 200