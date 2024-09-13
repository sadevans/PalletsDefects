# WPDD Система компьютерного зрения для поиска дефектов паллетов

## Репозиторий модели

Основной репозиторий проекта можно найти [тут](https://github.com/sadevans/BackWPDD).

### Используемые датасеты
- [Распознавание дефектов дерева (Large Scale Image Dataset of Wood Surface Defects)](https://www.kaggle.com/datasets/nomihsa965/large-scale-image-dataset-of-wood-surface-defects)

- [Распознавание паллет (pallet detection Computer Vision Project)](https://universe.roboflow.com/sundharesan-kumaresan/pallet-detection-ith6b)
- [Распознавание паллет (Computer Vision Project)]([https://universe.roboflow.com/sundharesan-kumaresan/pallet-detection-ith6b](https://universe.roboflow.com/palette/x-nbtav))
- [Самостоятельно собранный датасет + разметка](https://drive.google.com/drive/folders/1Z_Monpry0OlOtElsb2btXsvmj8nBJ3dB)

### Обученные модели

- [Модель детекции паллетов [YOLO]](https://drive.google.com/file/d/140vZOeVYqT5y5fGa84zMY_yNspFjeOwy/view?usp=sharing)
- [Модель детекции паллетов и дефектов паллетов [YOLO]](https://drive.google.com/file/d/1XsLvJ6dbJ4yyBbTFzl66V1UQbWQCSlKt/view?usp=sharing)
- [Модель детекции дефектов дерева [YOLO]](https://drive.google.com/file/d/10xUTNNiiNtDDcTXJC0v6w7nTDt6EmThU/view?usp=sharing). P.S. Плохо работает на реальных данных с паллетами
- [Модель классификации паллетов (в пленке / не в пленке) [MobileNetV2]](https://drive.google.com/file/d/1ZVC8dSctN0Y13qOBmPS7XZXf268Ze-FU/view?usp=sharing)
- [Модель классификации паллетов сбоку (заменить / не заменить) [ViT]](https://drive.google.com/file/d/1US2OXAzxvxiCNdqhHjbYOpCFdihkOqPj/view?usp=sharing)
- [Модель классификации паллетов снизу (заменить / не заменить) [ViT]](https://drive.google.com/file/d/1hRHMrUeWchxfvrNhMT_qEDqU1OLNAlHO/view?usp=sharing)

- ### Файловая архитектура проекта
  ```
  .
  ├── dataset_preparation_scripts
  |   ├── crop_objects.py
  |
  ├── jupyter_notebooks
  |   ├──wood_defects.ipynb
  |
  ├── pallet_processing
  |   ├──config
  |   |   |──__init__.py
  |   |   |──inference_config.py
  |   ├──models
  |   |   |──__init__.py
  |   |   |──MobileNetV2MembraneModel.py
  |   |   |──VitPalletModel.py
  |   ├──.env
  |   ├──__init__.py
  |   ├──full_model_inference.py
  |   ├──pipeline.py
  |   ├──settings.py
  ├──train_scripts
  |   ├──MobileNetV2
  |   |   ├──train_mobilenetv2_classifier.py
  |   ├──ViT
  |   |   ├──train_vit_classifier.py
  |   |   ├──inference_vit.py
  |   |_YOLO
  |   |   ├──train_yolo.py
  ├──.gitignore
  ├──README.md
  ├──download_models.py
  ├──setup.py
  ├──requirements.txt
  ```
Описание элементов файловой системы:
- `dataset_preparation_scripts` - директория, в которой находятся файлы подготовки датасета дефектов паллетов.
- `jupyter_notebooks` - директория, в которой находятся файлы формата Jupyter Notebook с обучением моделей.
- `pallet_processing` - python-package для работы с pipeline обработки изображения паллета.
- `pallet_processing/config` - python-package с конфигурациями.
- `pallet_processing/models` - python-package с инициализациями моделей.
- `pallet_processing/full_model_inference.py` - ключевой файл обработки моделей.
- `train_scripts` - директория, в которой находятся скрипты обучения моделей.
