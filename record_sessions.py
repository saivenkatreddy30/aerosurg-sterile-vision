import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from feature_pipeline import extract_surgical_features

# Initialize Hand Landmarker using the modern Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

session_label = input("Enter session name ('session_train' OR 'session_test'): ").strip()
os.makedirs(f"data/{session_label}", exist_ok=True)

gestures = ["pinch_zoom", "pan_flat", "next_slice", "emergency_lock"]
samples_per_gesture = 150

print(f"\n--- Starting data capture for: {session_label} ---")
for gesture in gestures:
    raw_bank, inv_bank = [], []
    print(f"\nPrepare gesture: [{gesture.upper()}]. Click into the webcam window and press 's' to start...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"Pose: {gesture.upper()} | Press 's'", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow("AeroSurg Data Studio", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    count = 0
    while count < samples_per_gesture:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            raw_f, inv_f = extract_surgical_features(detection_result.hand_landmarks[0])
            raw_bank.append(raw_f)
            inv_bank.append(inv_f)
            count += 1
            cv2.putText(frame, f"Recording {gesture}: {count}/{samples_per_gesture}", 
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("AeroSurg Data Studio", frame)
        cv2.waitKey(20)

    np.save(f"data/{session_label}/{gesture}_raw.npy", np.array(raw_bank))
    np.save(f"data/{session_label}/{gesture}_inv.npy", np.array(inv_bank))

cap.release()
cv2.destroyAllWindows()
print(f"Data saved to data/{session_label}/")