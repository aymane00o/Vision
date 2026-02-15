import cv2
import sys

# ──────────────────────────────────────────────
#  FACE DETECTION SCRIPT  |  Works in VS Code
#  Uses OpenCV's built-in Haar Cascade (no extra
#  weights file needed — works out of the box!)
# ──────────────────────────────────────────────

def load_detector():
    """Load the Haar Cascade face detector bundled with OpenCV."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("[ERROR] Could not load face cascade. Check your OpenCV install.")
        sys.exit(1)
    print("[OK] Face detector loaded.")
    return face_cascade


def detect_faces_in_image(image_path: str, face_cascade):
    """Run face detection on a still image and save the result."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not open image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # How much the image is scaled at each step
        minNeighbors=5,     # Higher = fewer false positives
        minSize=(40, 40)    # Minimum face size to detect
    )

    print(f"[INFO] Detected {len(faces)} face(s) in '{image_path}'")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output_path = "output_" + image_path.split("\\")[-1].split("/")[-1]
    cv2.imwrite(output_path, img)
    print(f"[SAVED] Result saved to: {output_path}")

    cv2.imshow("Face Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_faces_webcam(face_cascade):
    """Real-time face detection using your webcam. Press Q to quit."""
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Check it's connected and not in use.")
        return

    print("[INFO] Webcam started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show face count on screen
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        cv2.imshow("Face Detection - Webcam (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  MAIN  —  Choose your mode below
# ──────────────────────────────────────────────

if __name__ == "__main__":
    face_cascade = load_detector()

    # ▶ MODE 1: Webcam (real-time)
    detect_faces_webcam(face_cascade)

    # ▶ MODE 2: Detect from an image file
    # Uncomment the line below and replace with your image path:
    # detect_faces_in_image(r"D:\vision\photo.jpg", face_cascade)
    