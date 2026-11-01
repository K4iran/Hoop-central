from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHT = hf_hub_download(
    repo_id="GabrieleGiudici/E-BARD-detection-models",
    filename="BODD_yolov8n_0001.pt",
)

model = YOLO(WEIGHT)
names = model.names
if isinstance(names, list):
    class_names = [str(x) for x in names]
else:
    class_names = [str(names[i]) for i in sorted(names)]

matches = [i for i, name in enumerate(class_names) if "hoop" in name.lower()]
if len(matches) != 1:
    raise RuntimeError(f"Expected one hoop class, got names={class_names!r}")
hoop_class_id = matches[0]

exported = model.export(
    format="onnx",
    imgsz=704,
    opset=12,
    simplify=False,
    dynamic=False,
    nms=False,
    device="cpu",
)
exported_path = Path(exported)
target = OUT / "ebard-hoop.onnx"
shutil.copy2(exported_path, target)

metadata = {
    "model": "E-BARD YOLOv8n basketball detector",
    "source": "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models",
    "weights": "BODD_yolov8n_0001.pt",
    "license": "CC-BY-4.0",
    "inputSize": 704,
    "classNames": class_names,
    "hoopClassId": hoop_class_id,
}
(OUT / "ebard-hoop.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(f"Exported {target} ({target.stat().st_size:,} bytes); hoop class={hoop_class_id}")
