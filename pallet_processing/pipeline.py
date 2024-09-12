import torch
from torch import nn
from torchvision import models
from transformers import ViTModel
from pallet_processing.pallet_processing_models import *
from pallet_processing.settings import *

from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import time


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled_output)
        return logits
    
    

class InferencePipeline():
    def __init__(self, num_labels=2):
        super(InferencePipeline, self).__init__()

        self.pallet_defect_detection_model = YOLO(os.path.join(MODELS_PATH, YOLO_MODEL_PATH))
        self.bottom_classification_model = self.get_vit_model(BOTTOM_CLASSIF_NUM_LABELS, os.path.join(MODELS_PATH,BOTTOM_CLASSIF_MODEL_PATH))
        self.side_classification_model = self.get_vit_model(SIDE_CLASSIF_NUM_LABELS, os.path.join(MODELS_PATH,SIDE_CLASSIF_MODEL_PATH))
        self.packet_classification_model = self.get_mobilenet_model(PACKET_CLASSIF_NUM_LABELS, os.path.join(MODELS_PATH,PACKET_CLASSIF_MODEL_PATH))

        self.bottom_classification_model.eval()
        self.side_classification_model.eval()
        self.packet_classification_model.eval()

        self.bottom_classification_model.to(DEVICE)
        self.side_classification_model.to(DEVICE)
        self.packet_classification_model.to(DEVICE)


    def get_vit_model(self, num_labels, model_path):
        model = ViTForImageClassification(num_labels=num_labels)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return model


    def get_mobilenet_model(self, num_labels, model_path):
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.last_channel, num_labels)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return model


    # Трансформации для инференса
    def _get_inference_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


    # Загрузка изображения и его предобработка
    def _load_image(self, image_path, x1, y1, x2, y2):
        img = Image.open(image_path).convert("RGB")
        img = img.crop((x1, y1, x2, y2))
        transforms_pipeline = self._get_inference_transforms()
        return transforms_pipeline(img).unsqueeze(0)  # Добавляем batch dimension


    def _vit_predict_image(self, model, image_tensor, class_names, device):
        model.eval()
        with torch.no_grad():
            # start_time = time.time()

            image_tensor = image_tensor.to(device)

            outputs = model(image_tensor)

            _, predicted_class = torch.max(outputs, dim=1)

            # inference_time = time.time() - start_time

            return class_names[predicted_class.item()]


    def get_prediction(self, image_path, side='bottom'):

        response = {"replace_pallet": False, "defects_coords": [], "membrane": False, "pallet_coords": []}

        results = self.pallet_defect_detection_model(image_path, verbose=False, imgsz=1024)

        pallet_found = False
        defect_found = False
        x1, y1, x2, y2 = None, None, None, None
        defects_coords = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                label = int(box.cls[0])
                if label == 0:
                    pallet_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                elif label == 1:
                    defect_found = True
                    d_x1, d_y1, d_x2, d_y2 = map(int, box.xyxy[0])
                    defects_coords.append([d_x1, d_y1, d_x2, d_y2])

        # Если паллет не найден - заменить паллет

        if not pallet_found:
            response["replace_pallet"] = True
            return response

        response['pallet_coords'] = [x1, y1, x2, y2]

        # Если найдет дефект - заменить паллет

        if defect_found:
            response['defects_coords'] = defects_coords
            return response

        image_tensor = self._load_image(image_path, x1, y1, x2, y2).to(DEVICE)

        if side == 'bottom':

            with torch.no_grad():
                predicted_class = self._vit_predict_image(self.bottom_classification_model, image_tensor, DEFECT_CLASS_NAMES, DEVICE)

            # Если паллет на замену - заменить паллет

            if predicted_class == DEFECT_CLASS_NAMES[1]:
                response['replace_pallet'] = True

            return response

        else:

            with torch.no_grad():
                predicted_class = self._vit_predict_image(self.side_classification_model, image_tensor, DEFECT_CLASS_NAMES, DEVICE)

            # Если паллет на замену - заменить паллет
            if predicted_class == DEFECT_CLASS_NAMES[1]:
                response['replace_pallet'] = True

            with torch.no_grad():
                outputs = self.packet_classification_model(image_tensor)
                preds = torch.round(torch.sigmoid(outputs)).int().item()

            if preds == 1:
                response['membrane'] = True

            return response
        