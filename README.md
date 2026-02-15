# Vision
# 🚢 Vision Detection System

A professional computer vision project built with Python, OpenCV, and YOLOv8 — featuring **face detection**, **emotion recognition**, and **boat detection** with NVIDIA GPU support.

---

## 📸 Features

| Module | Description |
|--------|-------------|
| 😄 Face Detection | Real-time face detection using OpenCV Haar Cascades |
| 😢 Emotion Detection | Detects Smiling, Neutral, and Sad expressions with emoji overlay |
| 🚢 Boat Detection | High-accuracy boat detection using YOLOv8x + CUDA GPU |

---

## 🛠️ Requirements

### System
- Windows 10/11 (64-bit)
- Python **3.11** (recommended — PyTorch does not support Python 3.14)
- NVIDIA GPU with CUDA 12.8 support (optional but recommended)

### Python Packages
```
opencv-python
Pillow
ultralytics
torch
torchvision
torchaudio
numpy
```

---

## ⚡ Installation

### 1. Clone the repository
```bash
git clone https://github.com/aymane00o/vision-detection.git
cd vision-detection
```

### 2. Install Python 3.11
> ⚠️ PyTorch does NOT support Python 3.14. Use Python 3.11.
```bash
winget install Python.Python.3.11
```

### 3. Install PyTorch with CUDA (NVIDIA GPU)
```bash
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> For CPU only (no GPU):
> ```bash
> py -3.11 -m pip install torch torchvision torchaudio
> ```

### 4. Install remaining dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### 5. Verify GPU is detected
```bash
py -3.11 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```
Expected output:
```
CUDA: True
GPU: NVIDIA T1200 Laptop GPU
```

---

## 🚀 Usage

### 😄 Face Detection (Webcam)
```bash
py -3.11 face_detection.py
```
- Press `Q` to quit
- Press `S` to save a screenshot

### 😢 Face + Emotion Detection with Emoji
```bash
py -3.11 face_emotion_emoji.py
```
| Expression | Emoji | Box Color |
|-----------|-------|-----------|
| Smiling | 😄 | 🟢 Green |
| Neutral | 😐 | 🟡 Yellow |
| Sad | 😢 | 🔴 Red-Blue |

### 🚢 Boat Detection
```bash
py -3.11 boat_detection.py
```
> The YOLOv8 model (`yolov8x.pt`) will **auto-download** (~130MB) on first run.

**Switch modes** inside `boat_detection.py`:
```python
# Webcam (default)
detect_webcam(model)

# Video file
detect_video(model, r"D:\vision\boat_video.mp4")

# Image
detect_image(model, r"D:\vision\boat_photo.jpg")
```

---

## 📁 Project Structure

```
vision-detection/
│
├── face_detection.py          # Basic face detection (Haar Cascade)
├── face_emotion_emoji.py      # Face + emotion + emoji overlay
├── boat_detection.py          # YOLOv8 boat detection (GPU)
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignores weights, outputs, cache
└── README.md                  # This file
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit / close window |
| `S` | Save screenshot |
| `+` | Increase detection confidence |
| `-` | Decrease detection confidence |

---

## 🧠 Models Used

| Model | Type | Accuracy | Use Case |
|-------|------|----------|----------|
| `haarcascade_frontalface_default.xml` | Haar Cascade | ~85% | Face detection |
| `haarcascade_smile.xml` | Haar Cascade | ~75% | Smile detection |
| `yolov8x.pt` | YOLOv8 Extra-Large | ~95%+ | Boat detection |

---

## ⚠️ Known Issues

- **Python 3.14 not supported** — PyTorch requires Python 3.8–3.12
- **4GB VRAM limit** — `yolov8x` may be slow; switch to `yolov8m` in `boat_detection.py` if needed
- **Emotion detection accuracy** — depends on lighting and face angle; works best facing the camera directly

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Built With

- [OpenCV](https://opencv.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [Pillow](https://python-pillow.org/)
