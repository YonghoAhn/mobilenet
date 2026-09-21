# Hailo Edge Inference Experiments

Experimental repository for **edge AI inference on Hailo accelerators**, focused on fall detection and human-pose processing.

The project contains compiled Hailo models together with Python inference code used to test video-based emergency / fall-detection pipelines.

## Contents

- `inference.py` — video-based fall-detection inference with HailoRT
- `socket_inference_test.py` — integrated Hailo fall + pose pipeline with WebSocket streaming
- `mobilenet.hef` — compiled MobileNet-based model
- `yolov8m_pose.hef` — compiled YOLOv8m pose model
- `imagenet.hef` — additional compiled Hailo model used during deployment testing
- `*_compiled_model.html` — compiler / model-analysis reports
- `fall_detection_summary_*.json` — archived experiment summary

## Pipeline

The experimental flow combines:

1. video frame extraction / preprocessing
2. Hailo HEF inference
3. per-frame fall-probability estimation
4. YOLOv8 pose inference and keypoint decoding
5. sliding-window decision logic
6. optional WebSocket transmission to a monitoring dashboard

## Environment

The code was written for a HailoRT / Hailo Dataflow Compiler environment and expects the `hailo_platform` Python package to be available.

Typical additional dependencies include:

- OpenCV
- NumPy
- Pillow
- pandas
- tqdm
- websockets

Exact compatibility depends on the Hailo software stack and accelerator version used when the HEF files were compiled.

## Repository status

This repository is kept as an **experimental deployment snapshot**, not as a polished reusable library.

Generated runtime logs are intentionally excluded from the current tree. The compiled HEF files and compiler reports are retained because they document the actual accelerator deployment artifacts.
