from flask import Flask, request, jsonify
from retinaface.pre_trained_models import get_model
import cv2
import numpy as np

from deepfakedefender.infer import NetInference
from selfblended.infer import SelfBlended

app = Flask(__name__)
DEVICE = 'cpu'
self_blended_model = SelfBlended(DEVICE)
deep_fake_defender_model = NetInference(DEVICE)
print('Models loaded')


def extract_face(frame, face_detector_model, image_size=(380, 380)):
    faces = face_detector_model.predict_jsons(frame)

    if len(faces) == 0:
        print('No face is detected')
        return []
    cropped_faces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]['bbox']
        bbox = np.array([[x0, y0], [x1, y1]])
        cropped_faces.append(
            cv2.resize(crop_face(frame, bbox=bbox),
                       dsize=image_size).transpose((2, 0, 1)))
    return cropped_faces


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
    return img[y0_new:y1_new, x0_new:x1_new]


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_detector = get_model(model_name="resnet50_2020-07-20", max_size=380, device=DEVICE)
    face_detector.eval()
    face_list = extract_face(image, face_detector)
    return face_list


def _convert_to_image(file_storage):
    image_bytes = file_storage.read()
    # Convert raw bytes into a NumPy array
    image_np = np.frombuffer(image_bytes, np.uint8)
    # Decode the image as a NumPy array
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    print('Received image')

    file = request.files['file']
    image = _convert_to_image(file)
    face_list = preprocess_image(image)
    if len(face_list) == 0:
        return jsonify({'error': 'No face detected'})
    self_blended_pred = self_blended_model.infer(face_list)
    print(f'SelfBlended fakeness: {self_blended_pred}')
    deep_fake_defender_pred = deep_fake_defender_model.infer(image)
    print(f'DeepFakeDefender fakeness: {deep_fake_defender_pred}')
    return jsonify({'fakeness': round(float((self_blended_pred + deep_fake_defender_pred[0]) / 2), 4)})


if __name__ == '__main__':
    # serve(app, host='0.0.0.0', port=5090)
    app.run(host='0.0.0.0', port=5090, debug=True)
