import torch
from flask import Flask, request, jsonify
from model import Detector
from retinaface.pre_trained_models import get_model
from inference.preprocess import extract_face
import cv2
import numpy as np
from scipy.special import logit, expit

app = Flask(__name__)
device = torch.device("cpu")
model = Detector()
cnn_sd = torch.load('checkpoint_epoch_49_efficientnet-b4_bs_12_epoch_50_imgSize_380.pt', map_location=device, weights_only=True)
new_sd = {}
for k, v in cnn_sd.items():
    new_sd['net.' + k] = v
model.load_state_dict(new_sd)
model.eval()


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


def preprocess_image(file_storage):
    image_bytes = file_storage.read()
    # Convert raw bytes into a NumPy array
    image_np = np.frombuffer(image_bytes, np.uint8)
    # Decode the image as a NumPy array
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_detector = get_model(model_name="resnet50_2020-07-20", max_size=380, device=device)
    face_detector.eval()
    face_list = extract_face(image, face_detector)
    return face_list


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    face_list = preprocess_image(file)
    if len(face_list) == 0:
        return jsonify({'error': 'No face detected'})

    with torch.no_grad():
        img = torch.tensor(face_list).to(device).float() / 255
        pred = model(img).softmax(1)[:, 1].cpu().data.numpy().tolist()[0]
        y_logits = logit(pred)
        corrected = expit(3.563e-01 * y_logits + 5.780e-02)

    return jsonify({'fakeness': round(float(corrected), 4)})


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5090)