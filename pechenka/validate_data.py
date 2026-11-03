"""Check local YOLO image/label pairs before training."""

from collections import Counter
from pathlib import Path


CLASSES = ("game", "ads_close", "upgrade")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate(root: Path, require_all_classes: bool = False) -> Counter:
    counts: Counter = Counter()
    errors = []
    groups_by_split = {}
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = {
            path.stem: path for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        groups_by_split[split] = {
            stem.rsplit("_aug", 1)[0] if stem.rsplit("_aug", 1)[-1].isdigit() else stem
            for stem in images
        }
        for stem in sorted(images.keys() - labels.keys()):
            errors.append(f"{split}: image without label: {stem}")
        for stem in sorted(labels.keys() - images.keys()):
            errors.append(f"{split}: label without image: {stem}")
        for path in labels.values():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{path}:{number}: expected five YOLO values")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    if not 0 <= class_id < len(CLASSES):
                        raise ValueError("unknown class")
                    if not all(0 <= value <= 1 for value in coords):
                        raise ValueError("coordinates outside 0..1")
                    if coords[2] <= 0 or coords[3] <= 0:
                        raise ValueError("empty box")
                    counts[CLASSES[class_id]] += 1
                except ValueError as exc:
                    errors.append(f"{path}:{number}: {exc}")
    overlap = groups_by_split["train"] & groups_by_split["val"]
    if overlap:
        errors.append(
            f"{len(overlap)} screenshot groups occur in both train and val "
            "(augmented variants must stay in one split)"
        )
    if require_all_classes:
        for name in CLASSES:
            if not counts[name]:
                errors.append(f"no annotations for class {name}")
    print("Annotations:", dict(counts))
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors[:30])
                         + (f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""))
    return counts


if __name__ == "__main__":
    validate(Path(__file__).resolve().parent, require_all_classes=True)
