from flask import Flask, request, jsonify
from retinaface.pre_trained_models import get_model
import cv2
import numpy as np
import os
from functools import wraps

from deepfakedefender.infer import NetInference
from selfblended.infer import SelfBlended

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
DEVICE = 'cpu'
self_blended_model = SelfBlended(DEVICE)
deep_fake_defender_model = NetInference(DEVICE)
front_end_api_key = os.getenv("FRONTEND_API_KEY")
app.config['FRONTEND_API_KEY'] = front_end_api_key
print(f'Frontend API key: {front_end_api_key}')
print('Models loaded')


def extract_face(frame, face_detector_model, image_size=(380, 380)):
    faces = face_detector_model.predict_jsons(frame)

    if len(faces) == 0:
        print('No face is detected')
        return []

    highest_score = -1
    bbox_with_highest_score = None
    for f in faces:
        if 'bbox' not in f or len(f['bbox']) != 4:
            continue
        if f['score'] > highest_score:
            highest_score = f['score']
            bbox_with_highest_score = f['bbox']

    if bbox_with_highest_score is not None:
        print(f'Face detected with highest score: {highest_score}')
        x0, y0, x1, y1 = bbox_with_highest_score
        bbox = np.array([[x0, y0], [x1, y1]])
        cropped_face, x0_new, x1_new, y0_new, y1_new = crop_face(frame, bbox=bbox)
        return cv2.resize(cropped_face, dsize=image_size), x0_new, x1_new, y0_new, y1_new

    return None


def crop_face(img, bbox):
    height, weight = len(img), len(img[0])
    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    w = x1 - x0
    h = y1 - y0
    w0_margin = w / 2
    w1_margin = w / 2
    h0_margin = h / 2
    h1_margin = h / 2
    y0_new = int(max(0, y0 - h0_margin))
    y1_new = int(min(height, y1 + h1_margin + 1))
    x0_new = int(max(0, x0 - w0_margin))
    x1_new = int(min(weight, x1 + w1_margin + 1))
    return img[y0_new:y1_new, x0_new:x1_new], x0_new, x1_new, y0_new, y1_new


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_detector = get_model(model_name="resnet50_2020-07-20", max_size=380, device=DEVICE)
    face_detector.eval()
    return extract_face(image, face_detector)


def _convert_to_image(file_storage):
    image_bytes = file_storage.read()
    # Convert raw bytes into a NumPy array
    image_np = np.frombuffer(image_bytes, np.uint8)
    # Decode the image as a NumPy array
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key != app.config['FRONTEND_API_KEY']:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    print('Received image')
    file = request.files['file']
    image = _convert_to_image(file)
    face, x0_new, x1_new, y0_new, y1_new = preprocess_image(image)
    if face is None:
        return jsonify({'error': 'No face detected'})
    self_blended_pred = self_blended_model.infer([face.transpose((2, 0, 1))])
    print(f'SelfBlended fakeness: {self_blended_pred}')
    deep_fake_defender_pred = deep_fake_defender_model.infer(face)
    print(f'DeepFakeDefender fakeness: {deep_fake_defender_pred}')
    fakeness = round(float(max(deep_fake_defender_pred[0], self_blended_pred)), 4)

    return jsonify({'fakeness': fakeness,
                    'face_coordinates': {
                        'x0': x0_new,
                        'x1': x1_new,
                        'y0': y0_new,
                        'y1': y1_new
                    }})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy'
    }), 200


if __name__ == '__main__':
    app.run()
