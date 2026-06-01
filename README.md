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
