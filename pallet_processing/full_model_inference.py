from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image

# from inference_config import *
from pallet_processing.settings import *
from pallet_processing.pallet_processing_models import *

pallet_defect_detection_model = YOLO(os.path.join(MODELS_PATH, YOLO_MODEL_PATH))
bottom_classification_model = get_vit_model(BOTTOM_CLASSIF_NUM_LABELS, os.path.join(MODELS_PATH,BOTTOM_CLASSIF_MODEL_PATH))
side_classification_model = get_vit_model(SIDE_CLASSIF_NUM_LABELS, os.path.join(MODELS_PATH,SIDE_CLASSIF_MODEL_PATH))
packet_classification_model = get_mobilenet_model(PACKET_CLASSIF_NUM_LABELS, os.path.join(MODELS_PATH,PACKET_CLASSIF_MODEL_PATH))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

bottom_classification_model.eval()
side_classification_model.eval()
packet_classification_model.eval()

bottom_classification_model.to(device)
side_classification_model.to(device)
packet_classification_model.to(device)


# Трансформации для инференса
def _get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# Загрузка изображения и его предобработка
def _load_image(image_path, x1, y1, x2, y2):
    img = Image.open(image_path).convert("RGB")
    img = img.crop((x1, y1, x2, y2))
    transforms_pipeline = _get_inference_transforms()
    return transforms_pipeline(img).unsqueeze(0)  # Добавляем batch dimension


def _vit_predict_image(model, image_tensor, class_names, device):
    model.eval()
    with torch.no_grad():
        # start_time = time.time()

        image_tensor = image_tensor.to(device)

        outputs = model(image_tensor)

        _, predicted_class = torch.max(outputs, dim=1)

        # inference_time = time.time() - start_time

        return class_names[predicted_class.item()]


def get_prediction(image_path, side='bottom'):

    response = {"replace_pallet": False, "defects_coords": [], "membrane": False, "pallet_coords": []}

    results = pallet_defect_detection_model(image_path, verbose=False, imgsz=1024)

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

    image_tensor = _load_image(image_path, x1, y1, x2, y2).to(device)

    if side == 'bottom':

        with torch.no_grad():
            predicted_class = _vit_predict_image(bottom_classification_model, image_tensor, DEFECT_CLASS_NAMES, device)

        # Если паллет на замену - заменить паллет

        if predicted_class == DEFECT_CLASS_NAMES[1]:
            response['replace_pallet'] = True

        return response

    else:

        with torch.no_grad():
            predicted_class = _vit_predict_image(side_classification_model, image_tensor, DEFECT_CLASS_NAMES, device)

        # Если паллет на замену - заменить паллет
        if predicted_class == DEFECT_CLASS_NAMES[1]:
            response['replace_pallet'] = True

        with torch.no_grad():
            outputs = packet_classification_model(image_tensor)
            preds = torch.round(torch.sigmoid(outputs)).int().item()

        if preds == 1:
            response['membrane'] = True

        return response
    