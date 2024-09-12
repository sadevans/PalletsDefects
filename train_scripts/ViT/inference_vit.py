import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import argparse
import time


# Модель классификации ViT
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


# Трансформации для инференса
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# Загрузка изображения и его предобработка
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transforms_pipeline = get_inference_transforms()
    return transforms_pipeline(img).unsqueeze(0)  # Добавляем batch dimension


# Инференс изображения
def predict_image(model, image_tensor, class_names, device):
    model.eval()
    with torch.no_grad():
        start_time = time.time()

        image_tensor = image_tensor.to(device)

        outputs = model(image_tensor)

        _, predicted_class = torch.max(outputs, dim=1)

        inference_time = time.time() - start_time

        return class_names[predicted_class.item()], inference_time


# Основная функция инференса
def main():
    parser = argparse.ArgumentParser(description="Inference script for ViT image classification.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image for inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--class_names', type=str, nargs='+', required=True, help='List of class names')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='Device to run inference on (cpu or cuda)')
    args = parser.parse_args()

    # Проверяем наличие CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Загрузка модели
    model = ViTForImageClassification(num_labels=len(args.class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # Загрузка и предобработка изображения
    image_tensor = load_image(args.image_path)

    # Предсказание
    predicted_class, inference_time = predict_image(model, image_tensor, args.class_names, device)
    print(f"Predicted class: {predicted_class}")
    print(f"Inference time: {inference_time:.4f} seconds")


# Запуск основного скрипта
if __name__ == "__main__":
    main()
