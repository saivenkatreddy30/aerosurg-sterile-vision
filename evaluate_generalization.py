import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

gestures = ["pinch_zoom", "pan_flat", "next_slice", "emergency_lock"]

def load_dataset(session):
    raw_data, inv_data, targets = [], [], []
    for idx, gesture in enumerate(gestures):
        raw = np.load(f"data/{session}/{gesture}_raw.npy")
        inv = np.load(f"data/{session}/{gesture}_inv.npy")
        raw_data.append(raw)
        inv_data.append(inv)
        targets.extend([idx] * len(raw))
    return np.vstack(raw_data), np.vstack(inv_data), np.array(targets)

# Load data from both physical sessions
X_raw_train_all, X_inv_train_all, y_train_all = load_dataset("session_train")
X_raw_cross, X_inv_cross, y_cross = load_dataset("session_test")

# 80/20 train/test split for Same-Session evaluation
X_raw_tr, X_raw_same, y_tr, y_same = train_test_split(
    X_raw_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
)
X_inv_tr, X_inv_same, _, _ = train_test_split(
    X_inv_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
)

# Train on raw coordinates
model_raw = SVC(kernel='rbf', C=1.0, random_state=42)
model_raw.fit(X_raw_tr, y_tr)

# Train on engineered invariant features
model_inv = SVC(kernel='rbf', C=1.0, random_state=42)
model_inv.fit(X_inv_tr, y_tr)

# Save the trained invariant model for live deployment
joblib.dump(model_inv, "aerosurg_model.pkl")

# Compute 4-cell matrix
acc_raw_same = accuracy_score(y_same, model_raw.predict(X_raw_same)) * 100
acc_raw_cross = accuracy_score(y_cross, model_raw.predict(X_raw_cross)) * 100

acc_inv_same = accuracy_score(y_same, model_inv.predict(X_inv_same)) * 100
acc_inv_cross = accuracy_score(y_cross, model_inv.predict(X_inv_cross)) * 100

print("\n" + "="*62)
print("     AEROSURG: FOUR-CELL GENERALIZATION MATRIX")
print("="*62)
print(f"{'Feature Set':<22} | {'Same-Session Test':<18} | {'Cross-Session Test':<18}")
print("-" * 62)
print(f"{'Raw Coordinates (63)':<22} | {acc_raw_same:6.2f}%            | {acc_raw_cross:6.2f}%")
print(f"{'Invariant Features (7)':<22} | {acc_inv_same:6.2f}%            | {acc_inv_cross:6.2f}%")
print("="*62)