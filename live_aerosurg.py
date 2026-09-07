import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from feature_pipeline import extract_surgical_features

actions = ["PINCH_ZOOM (Measure)", "PAN_FLAT (Navigate)", "NEXT_SLICE (Scroll)", "EMERGENCY_LOCK (Freeze)"]
model = joblib.load("aerosurg_model.pkl")

# Modern MediaPipe Hand Landmarker Task
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    classifier_latency_ms = 0.0

    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            # Draw joint keypoints
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            _, inv_f = extract_surgical_features(landmarks)

            # Measure classifier latency independently
            t0 = time.perf_counter()
            pred = model.predict([inv_f])[0]
            classifier_latency_ms = (time.perf_counter() - t0) * 1000

            action_text = actions[pred]

            # Bounding box bounds
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
            y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)

            # Surgical overlay UI
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 220, 255), 2)
            cv2.rectangle(frame, (x_min, y_min - 32), (x_max, y_min), (0, 220, 255), cv2.FILLED)
            cv2.putText(frame, action_text, (x_min + 5, y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Frame-to-frame FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Display latency telemetry
    cv2.putText(frame, f"Pipeline FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Classifier Latency: {classifier_latency_ms:.2f} ms", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AeroSurg: Contactless Operating Room Interface", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()