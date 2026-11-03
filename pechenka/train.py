"""Validate local data, train YOLO, and export the best model to ONNX."""

import argparse
from pathlib import Path
from shutil import copy2

from validate_data import validate


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    validate(ROOT, require_all_classes=True)
    from ultralytics import YOLO

    base = ROOT / "yolov8n.pt"
    model = YOLO(str(base if base.is_file() else "yolov8n.pt"))
    model.train(
        data=str(ROOT / "data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        workers=0,
        name="cookie_three_class",
        patience=30,
        project=str(ROOT / "runs" / "detect"),
        exist_ok=True,
    )
    best = ROOT / "runs" / "detect" / "cookie_three_class" / "weights" / "best.pt"
    exported = Path(YOLO(str(best)).export(format="onnx", opset=12, simplify=False))
    target = ROOT / "weights" / "best.onnx"
    target.parent.mkdir(exist_ok=True)
    copy2(exported, target)
    print(f"Exported {target}")


if __name__ == "__main__":
    main()
