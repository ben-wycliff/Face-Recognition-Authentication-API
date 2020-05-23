from flask import Flask, jsonify,request
# from database.db import initialize_db
# from resources.admin import admin
import face_recognition, pickle
import numpy as np
from flask_cors import CORS

import base64

app = Flask(__name__)
CORS(app)

# app.config["MONGODB_SETTINGS"] = {"host": ""}  # database url

# initialize_db(app)

# app.register_blueprint(admin)


@app.route('/detect', methods=["POST"])
# @cross_origin()
def face_detect():
    try:
        body = request.get_json()
        datauri = body['datauri']
        datauri = datauri.partition(',')[2]
        img = open('testimage.png', 'wb')
        img.write(base64.b64decode(datauri))
        img.close()
        image = face_recognition.load_image_file('testimage.png')
        # with open('./resources/classifier.pkl', 'rb') as file:
            # model = pickle.load(file)
        model = pickle.load(open('model/classifier.sav', 'rb'))
        face_location = face_recognition.face_locations(image, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_location)
        if len(face_encodings) != 1:
            return {"user": "Unknown"}
        else:
            encodings_array = np.array([face_encodings[0]])
            prediction = model.predict(encodings_array)
            # prediction = model.predict([face_encodings])
            print(prediction)
            return {"user": prediction[0]}, 200
    except Exception:
        return "internal server error", 500

app.run(debug=True)
