import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report, fbeta_score
import seaborn as sns
import pandas as pd
from transformers import ViTModel

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# Проверка устройства
def get_device(cuda_index=0):
    return torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")


# Трансформации данных
def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# Загрузка данных
def load_data(data_dir, batch_size=64, valid_size=0.2):
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(data_dir, transform=val_transforms)

    num_data = len(train_data)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_data))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_loader, val_loader, train_data.classes


# Модель
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss


# Функция обучения
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100, device="cpu",
                model_path='vit.pth'):
    best_loss = float('inf')
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, loss = model(inputs, labels)
            if loss is None:
                loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, accuracy = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss <= best_loss and accuracy >= best_accuracy:
            best_loss = val_loss
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Saved state dict with accuracy: {accuracy:.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

    return model


# Оценка модели
def evaluate(model, loader, device="cpu"):
    model.eval()
    val_loss = 0.0
    # accuracy = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, loss = model(inputs, labels)

            if loss is None:
                loss = nn.CrossEntropyLoss()(outputs, labels)

            val_loss += loss.item()

            # test_output = outputs.argmax(1)
            # acc = (test_output == labels).sum().item() / len(labels)
            # acc = f1_score(test_output, labels)

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # accuracy += acc
        accuracy = fbeta_score(all_labels, all_preds, average='weighted', beta=1.2)

    return val_loss / len(loader), accuracy  # accuracy/len(loader)


# Отчет по классификации
def generate_classification_report(model, loader, class_names, device="cpu"):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, _ = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return pd.DataFrame(report).T


# Визуализация отчета
def plot_classification_report(report_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="coolwarm", cbar=False, ax=ax)
    plt.show()
    plt.savefig("report.png", bbox_inches='tight', dpi=300)


# Основная функция с argparse
def main():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model on a classification task.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where the dataset is located')
    parser.add_argument('--model_path', type=str, default='model_vit.pth', help='Path to save the trained model')
    parser.add_argument('--cuda_index', type=int, default=0, help='CUDA device index, default is 0')
    parser.add_argument('--valid_size', type=float, default=0.2, help='Validation dataset size (percentage)')

    args = parser.parse_args()

    # Получаем устройство
    device = get_device(args.cuda_index)

    # Загрузка данных
    train_loader, val_loader, class_names = load_data(args.data_dir, batch_size=args.batch_size,
                                                      valid_size=args.valid_size)

    # Инициализация модели
    model = ViTForImageClassification(2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)

    # Обучение модели
    model = train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=args.epochs, device=device,
                        model_path=args.model_path)

    # Сохранение модели
    model.load_state_dict(torch.load(args.model_path))

    # Генерация и визуализация отчета
    report_df = generate_classification_report(model, val_loader, class_names, device=device)
    plot_classification_report(report_df)


# Вызов основной функции
if __name__ == "__main__":
    main()
