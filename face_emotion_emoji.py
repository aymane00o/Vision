import cv2
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# ──────────────────────────────────────────────
#  FACE + EMOTION + EMOJI DETECTION SCRIPT
#  Shows real emoji overlaid on detected faces!
#  Smiling 😄 | Neutral 😐 | Sad 😢
# ──────────────────────────────────────────────

def load_detectors():
    base = cv2.data.haarcascades
    face_cascade  = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
    eye_cascade   = cv2.CascadeClassifier(base + "haarcascade_eye.xml")

    if face_cascade.empty() or smile_cascade.empty():
        print("[ERROR] Could not load cascades.")
        sys.exit(1)

    print("[OK] All detectors loaded.")
    return face_cascade, smile_cascade, eye_cascade


def make_emoji_image(emoji_char, size=80):
    """
    Render a real emoji character into a numpy image using Pillow.
    Returns an RGBA numpy array.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to load a system emoji font (Windows)
    font = None
    font_paths = [
        "C:/Windows/Fonts/seguiemj.ttf",   # Segoe UI Emoji (Windows 10/11)
        "C:/Windows/Fonts/seguisym.ttf",   # Segoe UI Symbol fallback
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, int(size * 0.8))
            break
        except:
            continue

    if font is None:
        # Fallback: draw colored circle if no emoji font found
        color_map = {"😄": (50, 220, 50), "😐": (50, 200, 255), "😢": (80, 80, 255)}
        color = color_map.get(emoji_char, (200, 200, 200))
        draw.ellipse([4, 4, size-4, size-4], fill=(*color, 230))
        draw.text((size//2, size//2), emoji_char[0], fill=(255,255,255,255),
                  anchor="mm")
        return np.array(img)

    # Draw emoji centered
    draw.text((size // 2, size // 2), emoji_char, font=font,
              embedded_color=True, anchor="mm")
    return np.array(img)


def overlay_emoji(frame, emoji_np, x, y, size):
    """
    Overlay a transparent RGBA emoji image onto the OpenCV BGR frame.
    Placed above the face bounding box.
    """
    # Resize emoji
    emoji_resized = cv2.resize(emoji_np, (size, size), interpolation=cv2.INTER_AREA)

    # Position: centered above the face box
    ex = x + (size // 2) - (size // 2)
    ey = max(0, y - size - 5)

    # Bounds check
    fh, fw = frame.shape[:2]
    ex = max(0, min(ex, fw - size))
    ey = max(0, min(ey, fh - size))

    # Blend using alpha channel
    roi = frame[ey:ey+size, ex:ex+size]
    if roi.shape[0] != size or roi.shape[1] != size:
        return  # Skip if out of bounds

    emoji_bgr  = cv2.cvtColor(emoji_resized[:,:,:3], cv2.COLOR_RGB2BGR)
    alpha      = emoji_resized[:,:,3:4].astype(np.float32) / 255.0
    blended    = (emoji_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
    frame[ey:ey+size, ex:ex+size] = blended


def detect_emotion(face_gray, smile_cascade, eye_cascade):
    h, w = face_gray.shape
    lower_face = face_gray[h//2:, :]
    upper_face = face_gray[:h//2, :]

    smiles = smile_cascade.detectMultiScale(
        lower_face, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
    )
    eyes = eye_cascade.detectMultiScale(
        upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    smile_detected = len(smiles) > 0
    eyes_open      = len(eyes) >= 1

    if smile_detected:
        return "Smiling",  "😄", (0, 220, 0),   "Happy"
    elif eyes_open:
        return "Neutral",  "😐", (0, 200, 255), "Neutral"
    else:
        return "Sad",      "😢", (80, 80, 255), "Sad"


def draw_face_box(frame, x, y, w, h, label, color):
    # Rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    cv2.rectangle(frame, (x, max(0, y - 32)), (x + w, y), color, -1)
    cv2.putText(frame, label, (x + 6, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Corner accents
    c, t = 14, 3
    pts = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (px,py),(dx,dy) in zip(pts,dirs):
        cv2.line(frame,(px,py),(px+c*dx,py),color,t)
        cv2.line(frame,(px,py),(px,py+c*dy),color,t)


def draw_hud(frame, face_count, emotion_counts):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Faces detected: {face_count}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    stats = f"Happy:{emotion_counts['Happy']}  Neutral:{emotion_counts['Neutral']}  Sad:{emotion_counts['Sad']}"
    cv2.putText(frame, stats, (w - 360, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.rectangle(overlay, (0, h-32), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "Q = Quit   S = Screenshot", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)


# ── Pre-render emoji images once at startup ────
EMOJI_CACHE = {}

def get_emoji(char, size=80):
    key = (char, size)
    if key not in EMOJI_CACHE:
        EMOJI_CACHE[key] = make_emoji_image(char, size)
    return EMOJI_CACHE[key]


# ──────────────────────────────────────────────
#  WEBCAM MODE
# ──────────────────────────────────────────────

def detect_webcam(face_cascade, smile_cascade, eye_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam running. Q = quit | S = screenshot")

    # Pre-render all emoji at startup
    for char in ["😄", "😐", "😢"]:
        get_emoji(char, 80)
    print("[OK] Emoji loaded.")

    shot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        emotion_counts = {"Happy": 0, "Neutral": 0, "Sad": 0}

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            label, emoji_char, color, key = detect_emotion(
                face_gray, smile_cascade, eye_cascade
            )
            emotion_counts[key] += 1

            # Draw box + label
            draw_face_box(frame, x, y, w, h, label, color)

            # ✨ Overlay emoji above face — size scales with face width
            emoji_size = max(50, min(w, 100))
            emoji_img  = get_emoji(emoji_char, emoji_size)
            overlay_emoji(frame, emoji_img, x, y, emoji_size)

        draw_hud(frame, len(faces), emotion_counts)
        cv2.imshow("Face & Emotion Detection  |  Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_count += 1
            path = f"D:\\vision\\screenshot_{shot_count}.jpg"
            cv2.imwrite(path, frame)
            print(f"[SAVED] {path}")

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  IMAGE MODE
# ──────────────────────────────────────────────

def detect_image(image_path, face_cascade, smile_cascade, eye_cascade):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot open: {image_path}")
        return

    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    emotion_counts = {"Happy": 0, "Neutral": 0, "Sad": 0}

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        label, emoji_char, color, key = detect_emotion(
            face_gray, smile_cascade, eye_cascade
        )
        emotion_counts[key] += 1
        draw_face_box(img, x, y, w, h, label, color)

        emoji_size = max(50, min(w, 100))
        emoji_img  = get_emoji(emoji_char, emoji_size)
        overlay_emoji(img, emoji_img, x, y, emoji_size)

    draw_hud(img, len(faces), emotion_counts)

    out = "D:\\vision\\output_emotion.jpg"
    cv2.imwrite(out, img)
    print(f"[SAVED] {out}")
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Install Pillow if needed:
    # pip install Pillow
    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    face_cascade, smile_cascade, eye_cascade = load_detectors()

    # ▶ MODE 1: Webcam — real-time with emoji
    detect_webcam(face_cascade, smile_cascade, eye_cascade)

    # ▶ MODE 2: Still image
    # detect_image(r"D:\vision\photo.jpg", face_cascade, smile_cascade, eye_cascade)
