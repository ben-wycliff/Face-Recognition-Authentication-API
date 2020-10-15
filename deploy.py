from flask import Flask, jsonify,request
# from database.db import initialize_db
# from resources.admin import admin
from .utils import generate_embeddings
from .utils import detect_face
import face_recognition, pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

import base64

app = Flask(__name__)
CORS(app)

df = pd.read_csv('./data/train-face-encodings.csv')
df.groupby("name").mean()


@app.route('/')
def index():
    return '<h1>Home</h1>'


@app.route('/detect', methods=["POST"])
def face_detect():
    # try:
        body = request.get_json()
        datauri = body['datauri']
        datauri = datauri.partition(',')[2]
        img = open('testimage.png', 'wb')
        img.write(base64.b64decode(datauri))
        img.close()
        image = face_recognition.load_image_file('testimage.png')

        model = pickle.load(open('model/classifier.sav', 'rb'))
        face_location = face_recognition.face_locations(image, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_location)
        if len(face_encodings) != 1:
            return {"user": "Unknown"}
        else:
            encodings_array = np.array([face_encodings[0]])
            prediction = model.predict(encodings_array)
            # prediction = model.predict([face_encodings])
            df = pd.read_csv('./data/train-face-encodings.csv')
            avg = df.groupby("name").mean()
            known = list(avg.loc[prediction[0]])
            known = known[-128: ]
            print(prediction)
            compare = face_recognition.compare_faces([known], face_encodings[0], tolerance=0.4)[0]
            if compare:
                return {"user": prediction[0]}, 200
            return {"user": "Unknown"}

@app.route('/login', methods=["POST"])
def face_recognition():
    body = request.get_json()
    datauri = body['datauri']
    datauri = datauri.partition(',')[2]
    img = open('userface.png', 'wb')
    img.write(base64.b64decode(datauri))
    img.close()
    face = detect_face('./userface.png')
    face_embeddings = generate_embeddings(face)
    # importing facenet-classifier
    classifier = pickle.load(open('models/classifier-facenet.sav', 'rb'))
    prediction = classifier.predict(face_embeddings)
    print(prediction)
    return {"prediction": prediction}