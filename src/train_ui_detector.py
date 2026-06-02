from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model for mobile UI element detection.")
    parser.add_argument("--data", default="data/ui.yaml", help="Path to the YOLO dataset YAML.")
    parser.add_argument(
        "--model",
        default="yolo11n.yaml",
        help="YOLO model YAML for training from scratch, or a .pt file for fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="mps", help="Training device, for example mps, cpu, or 0.")
    parser.add_argument("--project", default="runs/ui", help="Training output directory.")
    parser.add_argument("--name", default="ui-detector", help="Training run name.")
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights when supported. Leave unset for true scratch training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
