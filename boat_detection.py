import cv2
import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  🚢 PROFESSIONAL BOAT DETECTION SYSTEM
#  Powered by YOLOv8 + NVIDIA GPU
#  Detects: All boat types with maximum precision
#  Modes: Webcam | Video File | Image
# ══════════════════════════════════════════════════════════════

# ── Boat-related class IDs in COCO dataset ─────────────────────
# YOLOv8 trained on COCO knows these water vessel classes:
BOAT_CLASSES = {
    8:  "boat",
    9:  "ship / cargo vessel",
}

# Extended labels for display (maps to friendly names)
BOAT_LABELS = {
    "boat":               ("⛵ Boat",          (0,   220,  80)),
    "ship / cargo vessel":("🚢 Ship",          (0,   180, 255)),
}

DEFAULT_COLOR = (0, 200, 255)

# Output folder
OUTPUT_DIR = Path("D:/vision/boat_detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  SETUP & MODEL LOADING
# ──────────────────────────────────────────────────────────────

def check_dependencies():
    """Check all required packages are installed."""
    missing = []
    for pkg in ["ultralytics", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Run: pip install ultralytics torch torchvision")
        sys.exit(1)


def load_model(model_size="l"):
    """
    Load YOLOv8 model on GPU.
    Sizes: n=nano, s=small, m=medium, l=large, x=extra-large
    Use 'x' for maximum accuracy | 'n' for maximum speed
    """
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} | VRAM: {vram:.1f} GB")
    else:
        print("[WARN] No GPU detected — running on CPU (slower)")

    model_name = f"yolov8{model_size}.pt"
    print(f"[INFO] Loading {model_name} ... (auto-downloads if needed)")

    model = YOLO(model_name)
    model.to(device)

    print(f"[OK]  Model loaded on {device.upper()}")
    return model, device


# ──────────────────────────────────────────────────────────────
#  DETECTION CORE
# ──────────────────────────────────────────────────────────────

def run_detection(model, frame, conf_threshold=0.45, iou_threshold=0.45):
    """
    Run YOLOv8 inference and filter only boat-related detections.
    Returns list of (x1, y1, x2, y2, confidence, class_name)
    """
    results = model(
        frame,
        conf=conf_threshold,   # Minimum confidence to count as detection
        iou=iou_threshold,     # Overlap threshold for NMS
        verbose=False
    )

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Only include boat-related classes
            if cls_id in BOAT_CLASSES:
                class_name = BOAT_CLASSES[cls_id]
                detections.append((x1, y1, x2, y2, confidence, class_name))

    return detections


# ──────────────────────────────────────────────────────────────
#  DRAWING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def draw_detection(frame, x1, y1, x2, y2, confidence, class_name):
    """Draw a styled detection box with label and confidence."""
    label_info = BOAT_LABELS.get(class_name, (class_name, DEFAULT_COLOR))
    display_name, color = label_info

    w = x2 - x1
    h = y2 - y1

    # Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Confidence bar (below box)
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + w, y2 + 8),  (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 8), color, -1)

    # Label background
    label_text = f"{display_name}  {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Corner accents
    c, t = 16, 3
    corners = [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]
    for (cx, cy, dx, dy) in corners:
        cv2.line(frame, (cx, cy), (cx + c*dx, cy), color, t)
        cv2.line(frame, (cx, cy), (cx, cy + c*dy), color, t)


def draw_hud(frame, detections, fps=None, mode="LIVE"):
    """Draw top stats bar and bottom info bar."""
    fh, fw = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (fw, 52), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    boat_count = len(detections)
    avg_conf   = (sum(d[4] for d in detections) / boat_count * 100) if boat_count else 0

    cv2.putText(frame, f"BOAT DETECTOR", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(frame, f"Boats: {boat_count}", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 80), 2)

    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (fw - 130, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if boat_count:
        cv2.putText(frame, f"Avg conf: {avg_conf:.1f}%", (fw - 260, 35) if fps else (fw-200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Bottom bar
    cv2.rectangle(overlay, (0, fh - 34), (fw, fh), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    hint = "Q=Quit  S=Screenshot  +/-=Confidence" if mode == "LIVE" else "Processing..."
    cv2.putText(frame, hint, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)

    # Mode badge
    badge_color = (0, 150, 255) if mode == "LIVE" else (150, 0, 255)
    cv2.rectangle(frame, (fw//2 - 40, 8), (fw//2 + 40, 44), badge_color, -1)
    cv2.putText(frame, mode, (fw//2 - 28, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


# ──────────────────────────────────────────────────────────────
#  MODE 1: WEBCAM
# ──────────────────────────────────────────────────────────────

def detect_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    conf   = 0.45
    shot_n = 0
    prev_t = time.time()

    print("[INFO] Webcam started | Q=quit | S=screenshot | +/-=adjust confidence")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_detection(model, frame, conf_threshold=conf)

        for (x1, y1, x2, y2, confidence, cls) in detections:
            draw_detection(frame, x1, y1, x2, y2, confidence, cls)

        # FPS counter
        now  = time.time()
        fps  = 1.0 / (now - prev_t + 1e-9)
        prev_t = now

        draw_hud(frame, detections, fps=fps, mode="LIVE")
        cv2.imshow("🚢 Boat Detector | Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_n += 1
            path = OUTPUT_DIR / f"webcam_{datetime.now().strftime('%H%M%S')}_{shot_n}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[SAVED] {path}")
        elif key == ord('+') or key == ord('='):
            conf = min(0.95, conf + 0.05)
            print(f"[CONF] Threshold raised to {conf:.2f}")
        elif key == ord('-'):
            conf = max(0.10, conf - 0.05)
            print(f"[CONF] Threshold lowered to {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
#  MODE 2: VIDEO FILE
# ──────────────────────────────────────────────────────────────

def detect_video(model, video_path: str, conf=0.45):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = OUTPUT_DIR / f"output_{Path(video_path).stem}.mp4"
    writer   = cv2.VideoWriter(str(out_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps_in, (w, h))

    print(f"[INFO] Processing: {video_path}")
    print(f"[INFO] {total_frames} frames @ {fps_in:.1f} FPS")

    frame_n    = 0
    prev_t     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        detections = run_detection(model, frame, conf_threshold=conf)

        for (x1, y1, x2, y2, confidence, cls) in detections:
            draw_detection(frame, x1, y1, x2, y2, confidence, cls)

        now   = time.time()
        fps   = 1.0 / (now - prev_t + 1e-9)
        prev_t = now

        draw_hud(frame, detections, fps=fps, mode="VIDEO")
        writer.write(frame)

        # Progress
        pct = frame_n / total_frames * 100
        print(f"\r[{pct:5.1f}%] Frame {frame_n}/{total_frames} | "
              f"Boats: {len(detections)} | FPS: {fps:.1f}", end="")

        cv2.imshow("🚢 Boat Detector - Video | Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[SAVED] Output video: {out_path}")


# ──────────────────────────────────────────────────────────────
#  MODE 3: IMAGE FILE
# ──────────────────────────────────────────────────────────────

def detect_image(model, image_path: str, conf=0.45):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot open image: {image_path}")
        return

    print(f"[INFO] Analyzing image: {image_path}")
    detections = run_detection(model, img, conf_threshold=conf)

    for (x1, y1, x2, y2, confidence, cls) in detections:
        draw_detection(img, x1, y1, x2, y2, confidence, cls)

    draw_hud(img, detections, mode="IMAGE")

    print(f"[RESULT] Detected {len(detections)} boat(s):")
    for i, (x1, y1, x2, y2, confidence, cls) in enumerate(detections, 1):
        print(f"  {i}. {cls} — {confidence*100:.2f}% confidence at ({x1},{y1})")

    out_path = OUTPUT_DIR / f"result_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"[SAVED] {out_path}")

    cv2.imshow("🚢 Boat Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_dependencies()

    # ── Choose model size ──────────────────────────────────────
    # "n" = fastest  |  "s" = balanced  |  "m" = good
    # "l" = high accuracy  |  "x" = MAXIMUM accuracy (slower)
    MODEL_SIZE = "x"   # ← Change this (x = best accuracy)

    model, device = load_model(MODEL_SIZE)

    print("\n" + "═"*50)
    print("  🚢  BOAT DETECTION SYSTEM READY")
    print("═"*50)

    # ▶ MODE 1: Webcam — uncomment to use
    detect_webcam(model)

    # ▶ MODE 2: Video file — uncomment and set path
    # detect_video(model, r"D:\vision\boat_video.mp4")

    # ▶ MODE 3: Image — uncomment and set path
    # detect_image(model, r"D:\vision\boat_photo.jpg")
