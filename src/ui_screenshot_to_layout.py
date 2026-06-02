from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from xml.dom import minidom
from xml.etree import ElementTree as ET

from PIL import Image
from ultralytics import YOLO


ANDROID_NS = "http://schemas.android.com/apk/res/android"
ET.register_namespace("android", ANDROID_NS)


@dataclass
class UiElement:
    id: str
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    x_norm: float
    y_norm: float
    width_norm: float
    height_norm: float


def android_attr(name: str) -> str:
    return f"{{{ANDROID_NS}}}{name}"


def sanitize_id(label: str, index: int) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", label.strip().lower()).strip("_")
    if not normalized:
        normalized = "element"
    if normalized[0].isdigit():
        normalized = f"element_{normalized}"
    return f"{normalized}_{index:03d}"


def detect_elements(
    model_path: str,
    source_path: str,
    confidence: float,
    device: str | None,
    image_width: int,
    image_height: int,
) -> list[UiElement]:
    model = YOLO(model_path)
    results = model.predict(source=source_path, conf=confidence, device=device, verbose=False)
    if not results:
        return []

    names = results[0].names
    elements: list[UiElement] = []
    for index, box in enumerate(results[0].boxes, start=1):
        x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        label = names.get(int(box.cls[0]), str(int(box.cls[0])))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        elements.append(
            UiElement(
                id=sanitize_id(label, index),
                label=label,
                confidence=round(float(box.conf[0]), 4),
                x=x1,
                y=y1,
                width=width,
                height=height,
                x_norm=round(x1 / image_width, 6),
                y_norm=round(y1 / image_height, 6),
                width_norm=round(width / image_width, 6),
                height_norm=round(height / image_height, 6),
            )
        )

    return sorted(elements, key=lambda item: (item.y, item.x, item.height, item.width))


def widget_for_label(label: str) -> str:
    key = label.lower()
    if key in {"button", "btn"}:
        return "Button"
    if key in {"text", "label", "title", "subtitle", "paragraph"}:
        return "TextView"
    if key in {"input", "textfield", "edittext", "search", "textarea"}:
        return "EditText"
    if key in {"image", "img", "avatar", "photo", "banner"}:
        return "ImageView"
    if key in {"checkbox"}:
        return "CheckBox"
    if key in {"switch", "toggle"}:
        return "Switch"
    return "View"


def write_json(
    output_path: Path,
    source_path: str,
    image_width: int,
    image_height: int,
    elements: Iterable[UiElement],
) -> None:
    payload = {
        "source": source_path,
        "image": {"width": image_width, "height": image_height},
        "elements": [asdict(element) for element in elements],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_android_xml(
    output_path: Path,
    image_width: int,
    image_height: int,
    target_width_dp: int,
    elements: Iterable[UiElement],
) -> None:
    scale = target_width_dp / image_width
    target_height_dp = round(image_height * scale)

    root = ET.Element(
        "FrameLayout",
        {
            android_attr("layout_width"): f"{target_width_dp}dp",
            android_attr("layout_height"): f"{target_height_dp}dp",
        },
    )

    for element in elements:
        widget = widget_for_label(element.label)
        attrs = {
            android_attr("id"): f"@+id/{element.id}",
            android_attr("layout_width"): f"{max(1, round(element.width * scale))}dp",
            android_attr("layout_height"): f"{max(1, round(element.height * scale))}dp",
            android_attr("layout_marginStart"): f"{round(element.x * scale)}dp",
            android_attr("layout_marginTop"): f"{round(element.y * scale)}dp",
        }
        if widget == "TextView":
            attrs[android_attr("text")] = element.label
        elif widget == "Button":
            attrs[android_attr("text")] = element.label
        elif widget == "EditText":
            attrs[android_attr("hint")] = element.label
        elif widget == "ImageView":
            attrs[android_attr("scaleType")] = "centerCrop"
            attrs[android_attr("contentDescription")] = element.label
        root.append(ET.Element(widget, attrs))

    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="    ")
    output_path.write_text(pretty, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect mobile UI elements with YOLO and export layout files."
    )
    parser.add_argument("--source", required=True, help="Path to a phone UI screenshot.")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to a YOLO UI detection model.")
    parser.add_argument("--out", default="outputs/layout", help="Output path stem, without extension.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, mps, or 0.")
    parser.add_argument(
        "--target-width-dp",
        type=int,
        default=360,
        help="Android XML root width in dp. Coordinates are scaled from screenshot pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Screenshot not found: {source}")

    with Image.open(source) as image:
        image_width, image_height = image.size

    output_stem = Path(args.out)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    elements = detect_elements(
        model_path=args.model,
        source_path=str(source),
        confidence=args.conf,
        device=args.device,
        image_width=image_width,
        image_height=image_height,
    )

    json_path = output_stem.with_suffix(".json")
    xml_path = output_stem.with_suffix(".xml")
    write_json(json_path, str(source), image_width, image_height, elements)
    write_android_xml(xml_path, image_width, image_height, args.target_width_dp, elements)

    print(f"Detected {len(elements)} elements")
    print(f"Wrote {json_path}")
    print(f"Wrote {xml_path}")


if __name__ == "__main__":
    main()
