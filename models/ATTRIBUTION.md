# Hoop detector attribution

Hoop Central's optional automatic hoop locator uses the **E-BARD YOLOv8n basketball object-detection model**.

- Author: Gabriele Giudici
- Model repository: https://huggingface.co/GabrieleGiudici/E-BARD-detection-models
- Model weights: `BODD_yolov8n_0001.pt`
- Model licence stated by the author: CC BY 4.0
- Input resolution used for export: 704 × 704
- Browser export: ONNX, generated from the published checkpoint by `tools/export_ebard_hoop.py`

The published model card reports hoop precision of 0.993 and recall of 0.937 on its held-out E-BARD test split. These are the author's benchmark results, not a guarantee of performance on Hoop Central users' footage.

**Limitations:** the model has a single `Hoop` class; it does not separately detect the backboard, rim, or net. Its training data is broadcast basketball footage, so performance may be lower on practice-court videos, unusual camera angles, distant hoops, and shooting-machine setups. Hoop Central therefore uses the detection to locate a hoop region and still asks the user to tap the exact rim centre. Manual calibration remains available.

Please retain this attribution and the CC BY 4.0 licence when redistributing the converted model.
