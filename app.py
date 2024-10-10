import torch
from flask import Flask, request, jsonify
from model import Detector
from retinaface.pre_trained_models import get_model
from inference.preprocess import extract_face
import cv2
import numpy as np
from scipy.special import logit, expit

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Detector()
cnn_sd = torch.load('checkpoint_epoch_49_efficientnet-b4_bs_12_epoch_50_imgSize_380.pt', map_location=device)
new_sd = {}
for k, v in cnn_sd.items():
    new_sd['net.' + k] = v
model.load_state_dict(cnn_sd)
model = model.to(device)
model.eval()


def extract_face(frame, model, image_size=(380, 380)):
    faces = model.predict_jsons(frame)

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
    y0_new = max(0, y0 - h0_margin)
    y1_new = min(height, y1 + h1_margin + 1)
    x0_new = max(0, x0 - w0_margin)
    x1_new = min(weight, x1 + w1_margin + 1)

    return img[y0_new:y1_new, x0_new:x1_new]


def preprocess_image(uploaded_image):
    image = cv2.imread(uploaded_image)
    image = image.squeeze().numpy()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()
    face_list = extract_face(image, face_detector)
    return face_list


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_path = f'/tmp/{file.filename}'
    file.save(image_path)

    face_list = preprocess_image(image_path)

    if len(face_list) == 0:
        return jsonify({'error': 'No face detected'})

    with torch.no_grad():
        img = torch.tensor(face_list).to(device).float() / 255
        pred = model(img).softmax(1)[:, 1].cpu().data.numpy().tolist()
        y_logits = logit(pred)
        corrected = expit(0.2584 * y_logits + 0.3927)

    return jsonify({'fakeness': round(float(corrected[0]), 4)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5090, debug=True)
