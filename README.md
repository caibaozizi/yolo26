# yolo26

YOLO object detection workspace using Ultralytics.

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
yolo checks
```

## Object Detection

Run detection on an image:

```bash
yolo detect predict model=yolo11n.pt source=image.png device=mps
```

Run detection on a video:

```bash
yolo detect predict model=yolo11n.pt source=/path/to/video.mp4 device=mps
```

Run detection from the camera:

```bash
yolo detect predict model=yolo11n.pt source=0 device=mps
```

Prediction outputs are saved under `runs/detect/`.

## Python Usage

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="image.png",
    device="mps",
    conf=0.25,
    save=True,
)

for result in results:
    print(result.boxes)
```

Use `device=cpu` if `device=mps` is unavailable or unstable.

## UI Screenshot to Layout

This project can train a custom YOLO detector for mobile UI components and use
that detector to export layout drafts.

### Dataset

Put screenshots and labels in YOLO detection format:

```text
datasets/ui/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image needs a matching `.txt` label file with the same relative path:

```text
datasets/ui/images/train/home.png
datasets/ui/labels/train/home.txt
```

Each label line uses normalized YOLO box coordinates:

```text
class_id x_center y_center width height
```

Example:

```text
1 0.5000 0.8200 0.7200 0.0900
```

The class list is defined in [data/ui.yaml](data/ui.yaml):

```text
0 text
1 button
2 input
3 image
4 icon
5 navigation
6 tab
7 checkbox
8 switch
9 card
10 list_item
11 modal
12 toolbar
```

Before training, check the dataset:

```bash
python src/check_yolo_dataset.py --data data/ui.yaml
```

### Train From Scratch

Train a YOLO model from random initialization:

```bash
python src/train_ui_detector.py \
  --data data/ui.yaml \
  --model yolo11n.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device mps
```

The best trained model will be saved under:

```text
runs/ui/ui-detector/weights/best.pt
```

For a stronger baseline with pretrained weights, use a `.pt` model:

```bash
python src/train_ui_detector.py \
  --data data/ui.yaml \
  --model yolo11n.pt \
  --pretrained \
  --epochs 100 \
  --device mps
```

Fine-tuning from pretrained weights usually needs less data. True scratch
training needs substantially more labeled screenshots.

### Export Layout

Convert a phone UI screenshot into layout files:

```bash
python src/ui_screenshot_to_layout.py \
  --source /path/to/ui-screenshot.png \
  --model runs/ui/ui-detector/weights/best.pt \
  --out outputs/ui-layout \
  --device mps \
  --target-width-dp 360
```

The command writes:

- `outputs/ui-layout.json`: detected UI elements with pixel and normalized coordinates
- `outputs/ui-layout.xml`: Android `FrameLayout` draft using absolute margins
