"""
Tashqi validatsiya — BUSI'da o'qitilgan modelni BUS-BRA (Braziliya) datasetida test qilish.
Bu generalizatsiyani (modelning boshqa manbada ishlashini) isbotlaydi.

Ishlatish:  python external_validation.py
Natija:     external_metrics.json
"""
import json, csv
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = "breast_ai_model.onnx"
ROOT = Path("busbra_data/BUSBRA")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(path):
    a = np.array(Image.open(path).convert("RGB").resize((224, 224)), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def main():
    sess = ort.InferenceSession(MODEL)
    n_out = sess.get_outputs()[0].shape[-1]
    mal_idx = (n_out - 1) if isinstance(n_out, int) else 2  # malignant = oxirgi klass

    # Etiketlarni o'qish
    labels = {}
    with open(ROOT / "bus_data.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            labels[r["ID"]] = 1 if r["Pathology"].strip().lower() == "malignant" else 0

    y_true, y_score = [], []
    img_dir = ROOT / "Images"
    files = sorted(img_dir.glob("*.png"))
    for i, p in enumerate(files):
        stem = p.stem
        if stem not in labels:
            continue
        probs, _ = sess.run(None, {"image": preprocess(p)})
        y_true.append(labels[stem])
        y_score.append(float(probs[0][mal_idx]))
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}")

    y = np.array(y_true); s = np.array(y_score)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0

    # ROC + AUC
    ths = np.linspace(0, 1, 50); roc = []
    P, N = (y == 1).sum(), (y == 0).sum()
    for t in ths[::-1]:
        pr = s >= t
        roc.append({"fpr": round(float((pr & (y == 0)).sum() / N), 4),
                    "tpr": round(float((pr & (y == 1)).sum() / P), 4), "threshold": round(float(t), 3)})
    pts = sorted([(r["fpr"], r["tpr"]) for r in roc])
    auc = sum((pts[i][0] - pts[i-1][0]) * (pts[i][1] + pts[i-1][1]) / 2 for i in range(1, len(pts)))

    out = {
        "evaluated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "train_dataset": "BUSI (Al-Dhabyani 2020, Egypt)",
        "test_dataset": "BUS-BRA (Gomez-Flores 2024, Brazil) — biopsy-proven, different scanners",
        "n_total": len(y), "n_benign": int(N), "n_malignant": int(P),
        "accuracy": round(acc, 4), "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "precision": round(prec, 4), "f1": round(f1, 4), "auc": round(float(auc), 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc_curve": roc,
    }
    Path("external_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== TASHQI VALIDATSIYA (BUSI -> BUS-BRA) ===")
    for k in ["n_total", "accuracy", "sensitivity", "specificity", "f1", "auc"]:
        print(f"  {k:12s} {out[k]}")
    print("external_metrics.json yozildi")


if __name__ == "__main__":
    main()
