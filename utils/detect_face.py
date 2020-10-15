from mtcnn.mtcnn import MTCNN
from PIL import Image

def detect_face(filepath):
    print(filepath)
    required_size = (160, 160)
    image = Image.open(filepath)
    image = image.convert('RGB')
    image_pixels = np.asarray(image)
    detector = MTCNN()
    result = detector.detect_faces(image_pixels)
    if len(result) == 0:
        return "no face"
    x1, y1, width, height = result[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array