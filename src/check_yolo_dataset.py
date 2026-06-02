from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument("--data", default="data/ui.yaml", help="Path to the YOLO dataset YAML.")
    return parser.parse_args()


def resolve_dataset_root(data_yaml: Path, config: dict) -> Path:
    root = Path(config["path"])
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def check_split(root: Path, split: str, image_dir: str, class_count: int) -> tuple[int, list[str]]:
    errors: list[str] = []
    images_root = root / image_dir
    labels_root = root / image_dir.replace("images", "labels", 1)
    images = image_files(images_root)

    for image_path in images:
        relative = image_path.relative_to(images_root)
        label_path = (labels_root / relative).with_suffix(".txt")
        if not label_path.exists():
            errors.append(f"{split}: missing label for {image_path}")
            continue

        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                errors.append(f"{label_path}:{line_number}: expected 5 fields, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number}: fields must be numeric")
                continue
            if class_id < 0 or class_id >= class_count:
                errors.append(f"{label_path}:{line_number}: class id {class_id} is outside 0..{class_count - 1}")
            if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
                errors.append(f"{label_path}:{line_number}: box values must be normalized to 0..1")
            if width <= 0 or height <= 0:
                errors.append(f"{label_path}:{line_number}: box width and height must be positive")

    print(f"{split}: {len(images)} images")
    return len(images), errors


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config["names"]
    class_count = len(names)
    root = resolve_dataset_root(data_yaml, config)

    errors: list[str] = []
    for split in ("train", "val", "test"):
        image_dir = config.get(split)
        if image_dir:
            image_count, split_errors = check_split(root, split, image_dir, class_count)
            if split in {"train", "val"} and image_count == 0:
                split_errors.append(f"{split}: no images found")
            errors.extend(split_errors)

    if errors:
        print("\nDataset errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Dataset looks valid.")


if __name__ == "__main__":
    main()
