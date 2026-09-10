# AeroSurg: Sterile Touchless Surgical Display Controller

AeroSurg is a touchless computer vision interface engineered for sterile operating room environments where physical contact with keyboards and touchscreens is strictly prohibited. It enables hands-free radiology scan navigation using 4 clinical gestures: `PINCH_ZOOM`, `PAN_FLAT`, `NEXT_SLICE`, and `EMERGENCY_LOCK`.

This project benchmarks domain generalization under cross-session constraints, comparing raw MediaPipe coordinates against scale/rotation-invariant geometric features.

## 1. Feature Formulations
- **Raw Spatial Vector (63 dimensions):** Direct $(x, y, z)$ spatial coordinates from 21 MediaPipe hand landmarks.
- **Engineered Invariant Vector (7 dimensions):**
  - **Scale Normalization Anchor:** Euclidean distance between Landmark 0 (Wrist) and Landmark 9 (Middle MCP).
  - **Normalized Fingertip Distances (5 dimensions):** Distances from the wrist to each fingertip divided by the anchor scalar.
  - **Inter-finger Joint Angles (2 dimensions):** $\angle \text{Thumb-Wrist-Index}$ (pinch tracking) and $\angle \text{Index-Wrist-Middle}$ (slice navigation) computed via vector dot products.

## 2. Generalization Evaluation (Four-Cell Matrix)

| Feature Formulation | Same-Session Accuracy | Cross-Session Accuracy | Generalization Drop |
| :--- | :---: | :---: | :---: |
| **Raw Coordinates (63-D)** | 100.00% | 29.33% | -70.67% |
| **Invariant Features (7-D)** | 100.00% | 100.00% | **0.00%** |

### Explaining the Generalization Gap
When trained on raw spatial coordinates, the classifier overfits to the absolute pixel positions of the hand during training. In an independent cross-session environment (differing user distance, hand angle, and ambient lighting), the raw model undergoes complete domain collapse, dropping to 29.33% (near random chance). In contrast, our scale-normalized distances and inter-finger angles represent intrinsic hand geometry, retaining 100.00% operational accuracy across distinct sessions.

## 3. Latency Benchmarks
- **Classifier Inference Time:** ~0.35 ms to 0.50 ms (Scikit-Learn RBF-SVM)
- **Overall Pipeline Frame Rate:** ~28–32 FPS on standard CPU
- **Bottleneck Analysis:** Classifier inference latency is negligible (<1 ms). The primary pipeline bottleneck is MediaPipe's deep convolutional inference per frame (~28–32 ms).

## 4. Live Video Demonstration
- [Link to Real-Time Demo Video](https://drive.google.com/file/d/1YUUNe_2sA7-y5ROpU6RjdiAdjOY6fXVB/view?usp=sharing)

## 5. Setup & Running
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

python record_sessions.py
python evaluate_generalization.py
python live_aerosurg.py
