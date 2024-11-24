import torch
from selfblended.model import Detector
import os


class SelfBlended:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = Detector()
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
        model_path = os.path.join(current_dir, 'checkpoint_epoch_49_efficientnet-b4_bs_12_epoch_50_imgSize_380.pt')
        cnn_sd = torch.load(model_path, map_location=device,
                            weights_only=True)
        new_sd = {}
        for k, v in cnn_sd.items():
            new_sd['net.' + k] = v
        self.model.load_state_dict(new_sd)
        self.model.eval()

    def infer(self, face_list):
        with torch.no_grad():
            img = torch.tensor(face_list).to(self.device).float() / 255
            pred = self.model(img).softmax(1)[:, 1].cpu().data.numpy().tolist()[0]
            return pred
