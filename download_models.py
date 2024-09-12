import gdown
import os
from pallet_processing.settings import MODELS_PATH

def download_models(models):
    for id in models.keys():
        print(models[id])
        gdown.download(models[id]['url'], os.path.join(MODELS_PATH, models[id]['name']), quiet=False)


if __name__ == "__main__":
    yolo_url = 'https://drive.google.com/uc?id=1XsLvJ6dbJ4yyBbTFzl66V1UQbWQCSlKt'
    vit_side_url = 'https://drive.google.com/uc?id=1US2OXAzxvxiCNdqhHjbYOpCFdihkOqPj'
    vit_bottom_url = 'https://drive.google.com/uc?id=1hRHMrUeWchxfvrNhMT_qEDqU1OLNAlHO'
    mobilenet_url = 'https://drive.google.com/uc?id=1ZVC8dSctN0Y13qOBmPS7XZXf268Ze-FU'


    models_to_download = {
        1: {'url': yolo_url, 'name': 'pallets_plus_defects.pt'},
        2: {'url': vit_side_url, 'name': 'vit_side_v2.pth'},
        3: {'url': vit_bottom_url, 'name': 'vit_bottom_v2.pth'},
        4: {'url': mobilenet_url, 'name': 'mobilenet_v2_binary_classification_packet.pth'},
    }

    download_models(models_to_download)
