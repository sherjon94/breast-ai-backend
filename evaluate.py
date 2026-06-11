"""
Breast AI — Model baholash skripti
BUSI dataseti bilan modelni test qilib metrics.json yaratadi.

Ishlatish:
    python evaluate.py <dataset_papka>

Dataset tuzilishi (BUSI standarti):
    dataset/
      benign/      *.png  (mask fayllarsiz, "_mask" nomlilar tashlab yuboriladi)
      malignant/   *.png

Natija: metrics.json — backend /api/metrics endpointi orqali frontendda ko'rsatiladi.
"""

import sys
import json
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = Path(__file__).parent / "breast_ai_model.onnx"
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - IMG_MEAN) / IMG_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def collect_images(root: Path, cls: str):
    folder = root / cls
    if not folder.exists():
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files += [p for p in folder.glob(ext) if "_mask" not in p.stem.lower()]
    return sorted(files)


def roc_curve_points(y_true, y_score, n_points=50):
    """ROC nuqtalari (FPR, TPR) — threshold bo'yicha"""
    thresholds = np.linspace(0, 1, n_points)
    points = []
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    for t in thresholds[::-1]:
        pred = y_score >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        points.append({"fpr": round(fpr, 4), "tpr": round(tpr, 4), "threshold": round(float(t), 3)})
    return points


def auc_from_roc(points):
    """Trapezoid usuli bilan AUC"""
    pts = sorted([(p["fpr"], p["tpr"]) for p in points])
    auc = 0.0
    for i in range(1, len(pts)):
        auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2
    return round(auc, 4)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Papka topilmadi: {root}")
        sys.exit(1)

    session = ort.InferenceSession(str(MODEL_PATH))

    benign_files = collect_images(root, "benign")
    malignant_files = collect_images(root, "malignant")
    print(f"Benign: {len(benign_files)} ta, Malignant: {len(malignant_files)} ta rasm")
    if not benign_files or not malignant_files:
        print("benign/ va malignant/ papkalarida rasm topilmadi!")
        sys.exit(1)

    y_true, y_score = [], []
    for label, files in ((0, benign_files), (1, malignant_files)):
        for i, f in enumerate(files):
            try:
                probs, _ = session.run(None, {"image": preprocess(f)})
                y_true.append(label)
                y_score.append(float(probs[0][1]))  # malignant ehtimoli
            except Exception as e:
                print(f"  Xato {f.name}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  {'malignant' if label else 'benign'}: {i+1}/{len(files)}")

    y_true_a = np.array(y_true)
    y_pred_a = (np.array(y_score) >= 0.5).astype(int)

    tp = int(((y_pred_a == 1) & (y_true_a == 1)).sum())
    tn = int(((y_pred_a == 0) & (y_true_a == 0)).sum())
    fp = int(((y_pred_a == 1) & (y_true_a == 0)).sum())
    fn = int(((y_pred_a == 0) & (y_true_a == 1)).sum())

    accuracy    = (tp + tn) / len(y_true) if y_true else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0  # recall
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision   = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0

    roc = roc_curve_points(y_true, y_score)
    auc = auc_from_roc(roc)

    metrics = {
        "evaluated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "dataset": str(root),
        "n_benign": len(benign_files),
        "n_malignant": len(malignant_files),
        "n_total": len(y_true),
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "auc": auc,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc_curve": roc,
    }

    out = Path(__file__).parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== NATIJALAR ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"\nmetrics.json yaratildi: {out}")
    print("Endi uni Render'ga deploy qiling (git add metrics.json) — frontend avtomatik ko'rsatadi.")


if __name__ == "__main__":
    main()
