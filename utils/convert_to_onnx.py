import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path to the model")
    parser.add_argument("--format", type=str, default="onnx", help="Expected export format")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.export(format=args.format)
