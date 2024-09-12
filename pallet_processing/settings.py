import os
from dotenv import load_dotenv
import torch

if not 'COMPILED' in os.environ:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH")
BOTTOM_CLASSIF_MODEL_PATH = os.environ.get("BOTTOM_CLASSIF_MODEL_PATH")
SIDE_CLASSIF_MODEL_PATH = os.environ.get("SIDE_CLASSIF_MODEL_PATH")
PACKET_CLASSIF_MODEL_PATH = os.environ.get("PACKET_CLASSIF_MODEL_PATH")
MODELS_PATH = os.environ.get("MODELS_PATH")

BOTTOM_CLASSIF_NUM_LABELS = 2
SIDE_CLASSIF_NUM_LABELS = 2
PACKET_CLASSIF_NUM_LABELS = 1

DEFECT_CLASS_NAMES = ['good', 'bad']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')