import torch
from torch import nn
from torchvision import models
from transformers import ViTModel


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


def get_vit_model(num_labels, model_path):
    model = ViTForImageClassification(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    return model


def get_mobilenet_model(num_labels, model_path):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_labels)
    model.load_state_dict(torch.load(model_path))
    return model
