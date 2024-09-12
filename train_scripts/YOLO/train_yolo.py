from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    # Get arguments to argparse.Namespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to yaml file")
    parser.add_argument("--model", type=str, default='yolov9s.pt', help="Model version format yolo<version>.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Num of batches")
    parser.add_argument("--workers", type=int, default=10, help="Num of workers")
    parser.add_argument("--epochs", type=int, default=50, help="Train epochs count")
    parser.add_argument("--freeze", type=int, default=10, help="Train epochs count")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument('--persp', type=float, default=0.0)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--optim', type=str, default=None)

    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                          device="cuda:0", lr0=args.lr0, lrf=args.lrf, mosaic=args.mosaic, freeze=args.freeze)
    