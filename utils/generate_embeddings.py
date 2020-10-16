from numpy import expand_dims
from keras.models import load_model
import sys
sys.path.append("lib/")
from inception_resnet_v1 import *

model = load_model("models/keras/model/facenet_keras.h5")

def generate_embeddings(face):
    face_pixels = face.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # transforming face into one sample.
    samples = expand_dims(face_pixels, axis=0)

    # making prediction to get embedding
    embeddings_arr = model.predict(samples)
    return embeddings_arr