import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
from model import Detector
from retinaface.pre_trained_models import get_model
from inference.preprocess import extract_face, crop_face
import io
import cv2

app = Flask(__name__)

model=Detector()
model=model.cuda()
cnn_sd=torch.load('model.pth', map_location=torch.device('cpu'))
model.load_state_dict(cnn_sd)
model.eval()

def preprocess_image(image):
    # add code for preprocessing
    image = cv2.imread(example_fake_image)

    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype('double')

    face_detector = get_model("resnet50_2020-07-20", max_size=max(image.shape),device='cuda:0')
    face_detector.eval()
    face_list=extract_face(image,face_detector, (128, 128))
    return face_list

    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img_bytes = file.read()

    face_list = preprocess_image(img_bytes)

    with torch.no_grad():
       img=torch.tensor(face_list).to('cuda:0').float()/255
       pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    return jsonify({'fakeness': round(max(pred), 4)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)