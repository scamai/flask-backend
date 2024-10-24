from .network import MFF_MoE
import torch.nn as nn
from torchvision import transforms
import torch
from PIL import Image
import numpy as np


class NetInference:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.net = MFF_MoE(pretrained=False, device=self.device)
        # need to change the path to the correct path
        self.net.load(path='/Users/david.xu/code/flask-backend/deepfakedefender/')
        if self.device == 'cuda':
            self.net = nn.DataParallel(self.net).to(self.device)
        self.net.eval()
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512), antialias=True),
        ])

    def infer(self, image):
        x = image[..., ::-1]
        x = Image.fromarray(np.uint8(x))
        x = self.transform_val(x).unsqueeze(0).to(self.device)
        pred = self.net(x)
        pred = pred.detach().cpu().numpy()
        return pred
