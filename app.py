import torch
from flask import Flask, request, jsonify
from model import Detector
from retinaface.pre_trained_models import get_model
from inference.preprocess import extract_face, crop_face
import cv2

app = Flask(__name__)

device = torch.device('cpu')
model=Detector()
cnn_sd=torch.load('model.pth', map_location=device)
model.load_state_dict(cnn_sd)
model = model.to(device)
model.eval()

def preprocess_image(uploadedImage):
    # add code for preprocessing
    image = cv2.imread(uploadedImage)

    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype('double')

    face_detector = get_model("resnet50_2020-07-20", max_size=max(image.shape),device=device)
    face_detector.eval()
    face_list=extract_face(image,face_detector, (128, 128))
    return face_list

    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image_path = f'/tmp/{file.filename}'
    file.save(image_path)

    face_list = preprocess_image(image_path)

    with torch.no_grad():
       img=torch.tensor(face_list).to(device).float()/255
       pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    return jsonify({'fakeness': round(max(pred), 4)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5090, debug=True)

    app.run(host='0.0.0.0', port=5090, debug=True)