# Cookie clicker detector

Windows clicker using `weights/best.onnx`. The model has exactly three classes:
`game`, `ads_close`, `upgrade` (IDs 0, 1, 2). An ad close button has priority,
then an available upgrade, then the main cookie. The model sometimes labels a
whole side banner as `ads_close`; the app only clicks small close-button boxes.
Some modal close buttons are emitted as `upgrade`; when the overlay hides the
cookie, those small boxes are handled as ad close buttons before shop checks.
For Yandex's white modal ad card, the app also finds the grey round `×` button
directly in the screenshot, so closing still works when ONNX returns no boxes.
Before buying, the app checks the shop's price state in the screenshot: green
prices are affordable, red prices are skipped. It randomly picks one affordable
row and keeps farming when none is available. The pixel check follows the
default Cookie Clicker Old shop colors shown in the local screenshots.

## Run

Install Python 3.12, then in PowerShell:

```
в app.py на 158 строке сделать
return pyautogui.keyDown("space")


python3 -m venv venv
source venv/bin/activate
pip install opencv-python
pip install pyautogui
pip install mss
pip install onnxruntime
python app.py

```

**Space** pauses or resumes. **Ctrl+C** exits. Moving the mouse to a screen
corner activates PyAutoGUI's fail-safe. Use `--monitor 0` for all displays or
`--monitor 1` (default) for the first display. The app fails with a clear error
if the ONNX model is missing or has the wrong classes.

## Retrain locally

Training images and YOLO labels stay outside Git. Store matching pairs under
`images/{train,val}` and `labels/{train,val}` using IDs from `data.yaml`.
`prepare_dataset.ps1` rebuilds a group-aware 80/20 split while preserving
unpaired files in `dataset_unmatched`.

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt
./prepare_dataset.ps1
.venv\Scripts\python validate_data.py
.venv\Scripts\python train.py --epochs 100
```

The training script exports its best checkpoint to `weights/best.onnx`. Only
this deployment model is tracked in Git; local datasets, training runs, and
Python environments are ignored.
# PechenkaFh
