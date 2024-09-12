import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import fbeta_score, confusion_matrix
import copy
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Функция для парсинга аргументов
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for binary classification with MobileNetV2")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--save_path', type=str, default='mobilenet_v2_binary_classification.pth',
                        help='Path to save the trained model')
    return parser.parse_args()


# Главная функция
def main():
    args = parse_args()

    # Гиперпараметры
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs

    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Трансформации для данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Загрузка датасетов
    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Модель: MobileNetV2 (Large)
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    # Изменяем классификационный слой на бинарную классификацию
    model.classifier[1] = nn.Linear(model.last_channel, 1)  # 1 выход для бинарной классификации
    model = model.to(device)

    # Функция потерь и оптимизатор
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    trained_model = train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

    # Сохранение обученной модели
    torch.save(trained_model.state_dict(), args.save_path)
    print(f'Model saved to {args.save_path}')

    paint_confusion_matrix(model=trained_model, loader=val_loader, device=device)


def paint_confusion_matrix(model, loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = torch.round(torch.sigmoid(outputs))  # Преобразуем вероятности в классы 0 или 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Рассчет F1-метрики
    conf_matrix = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


# Функция для обучения модели
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    best_model_wts = None
    best_f1 = 0.0

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Валидация
        val_f1 = validate_model(model, val_loader, device)
        print(f'Validation F1: {val_f1:.4f}')

        # Сохранение лучших весов
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'Best model updated at epoch {epoch + 1}')

    # Загружаем лучшие веса модели
    model.load_state_dict(best_model_wts)
    return model


# Функция для валидации
def validate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = torch.round(torch.sigmoid(outputs))  # Преобразуем вероятности в классы 0 или 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Рассчет F1-метрики
    val_f1 = fbeta_score(all_labels, all_preds, beta=1.5)
    return val_f1


if __name__ == '__main__':
    main()
