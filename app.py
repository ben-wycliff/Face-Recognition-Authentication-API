from flask import Flask, jsonify,request
# from database.db import initialize_db
# from resources.admin import admin
import face_recognition, pickle
import numpy as np
app = Flask(__name__)

# app.config["MONGODB_SETTINGS"] = {"host": ""}  # database url

# initialize_db(app)

# app.register_blueprint(admin)


@app.route('/detect', methods=["POST"])
def face_detect():
    body = request.get_json()
    print(body)
    image = face_recognition.load_image_file('./resources/image.jpg')
    # with open('./resources/classifier.pkl', 'rb') as file:
        # model = pickle.load(file)
    model = pickle.load(open('./resources/classifier.sav', 'rb'))
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


app.run(debug=True)
