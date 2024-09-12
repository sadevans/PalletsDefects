import torch
from torch import nn
from torchvision import models


def get_mobilenet_model(num_labels, model_path):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_labels)
    model.load_state_dict(torch.load(model_path))
    return model
