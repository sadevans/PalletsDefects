import os
import cv2
import argparse
from ultralytics import YOLO
from shapely.geometry import box


# Функция для добавления аргументов
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 detection and cropping")

    # Добавляем аргументы для путей к моделям и папкам
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели YOLOv5')
    parser.add_argument('--images_path', type=str, required=True, help='Путь к папке с изображениями')
    parser.add_argument('--labels_path', type=str, required=True, help='Путь к папке с метками')
    parser.add_argument('--damaged_dir', type=str, default="damaged", help='Путь к папке для повреждённых объектов')
    parser.add_argument('--not_damaged_dir', type=str, default="not_damaged",
                        help='Путь к папке для неповреждённых объектов')

    return parser.parse_args()


# Функция для проверки пересечения двух bbox
def check_overlap(bbox1, bbox2):
    box1 = box(bbox1[0], bbox1[1], bbox1[2], bbox1[3])
    box2 = box(bbox2[0], bbox2[1], bbox2[2], bbox2[3])
    return box1.intersects(box2)


def main():
    # Парсим аргументы
    args = parse_args()

    # Инициализация модели YOLOv5
    model = YOLO(args.model)

    # Создаем папки для сохранения результатов, если их нет
    os.makedirs(args.damaged_dir, exist_ok=True)
    os.makedirs(args.not_damaged_dir, exist_ok=True)

    # Проходим по изображениям датасета
    for image_name in os.listdir(args.images_path):
        image_path = os.path.join(args.images_path, image_name)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]  # Получаем размеры изображения для денормализации координат

        # Детекция объектов класса 0
        results = model.predict(image)  # Используем predict для детекции
        detected_bboxes_class_0 = []

        for bbox in results[0].boxes.xyxyn:  # Получаем нормализованные координаты bbox
            cls = int(bbox[-1])  # Класс объекта
            if cls == 0:  # Проверка на класс 0
                # Денормализуем координаты в пиксели
                x_min = bbox[0] * width
                y_min = bbox[1] * height
                x_max = bbox[2] * width
                y_max = bbox[3] * height
                detected_bboxes_class_0.append([x_min, y_min, x_max, y_max])

        # Путь к файлу меток (предполагается формат аннотаций YOLO)
        label_name = image_name.replace(".jpeg", ".txt")  # Соответствующее имя файла меток
        label_path = os.path.join(args.labels_path, label_name)

        if not os.path.exists(label_path):
            print(f"Метка для {image_name} не найдена, пропускаем.")
            continue

        # Чтение меток классов
        with open(label_path, "r") as file:
            annotations = file.readlines()

        bboxes_class_1 = []

        for line in annotations:
            class_id, x_center, y_center, width_bbox, height_bbox = map(float, line.strip().split())
            if class_id == 1:
                # Денормализуем координаты bbox из YOLO формата в пиксели
                x_min = (x_center - width_bbox / 2) * width
                y_min = (y_center - height_bbox / 2) * height
                x_max = (x_center + width_bbox / 2) * width
                y_max = (y_center + height_bbox / 2) * height
                bboxes_class_1.append([x_min, y_min, x_max, y_max])

        # Проверка пересечений и сохранение обрезков
        for bbox_0 in detected_bboxes_class_0:
            is_damaged = False

            for bbox_1 in bboxes_class_1:
                if check_overlap(bbox_0, bbox_1):
                    is_damaged = True
                    break

            cropped_image = image[int(bbox_0[1]):int(bbox_0[3]), int(bbox_0[0]):int(bbox_0[2])]

            # Сохраняем обрезанное изображение в соответствующую папку
            if is_damaged:
                save_path = os.path.join(args.damaged_dir, image_name)
            else:
                save_path = os.path.join(args.not_damaged_dir, image_name)

            cv2.imwrite(save_path, cropped_image)


if __name__ == "__main__":
    main()