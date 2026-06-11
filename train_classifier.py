"""
Breast AI — Klassifikator o'qitish (haqiqiy BUSI ma'lumotida)
MobileNetV3-Small (ImageNet pretrained) -> benign/malignant 2-klass

Ishlatish:
    python train_classifier.py busi_data/Dataset_BUSI_with_GT

Natija:
    breast_ai_model.onnx  — yangi, haqiqiy model (eski interfeysga mos)
    metrics.json          — TEST to'plamida (ajratilgan) halol metrikalar

Ilmiy to'g'rilik: stratified 70/15/15 split, fixed seed. Test to'plami
o'qitishda ko'rilmaydi — metrics.json haqiqiy generalizatsiyani aks ettiradi.
"""
import sys, json, random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, (torch.get_num_threads())))

CLASSES = ["benign", "malignant"]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def collect(root: Path):
    samples = []
    for idx, c in enumerate(CLASSES):
        for p in (root / c).glob("*.png"):
            if "mask" in p.stem.lower():
                continue
            samples.append((str(p), idx))
    random.shuffle(samples)
    return samples


def split(samples):
    """Stratified 70/15/15"""
    by_cls = {0: [], 1: []}
    for s in samples:
        by_cls[s[1]].append(s)
    tr, va, te = [], [], []
    for c, items in by_cls.items():
        n = len(items)
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        tr += items[:n_tr]
        va += items[n_tr:n_tr + n_va]
        te += items[n_tr + n_va:]
    random.shuffle(tr)
    return tr, va, te


class BUSIDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


class BreastClassifier(nn.Module):
    """MobileNetV3-Small + 2-klass head. ONNX: class_probs[1,2], birads_score[1,1]"""
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_feat = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_feat, 2)

    def forward(self, x):
        logits = self.backbone(x)
        probs = F.softmax(logits, dim=1)
        birads_score = probs[:, 1:2]  # malignant ehtimoli = BI-RADS xavf signali
        return probs, birads_score


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    all_scores, all_labels = [], []
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        probs, _ = model(imgs)
        logits = torch.log(probs + 1e-9)
        loss = criterion(logits, labels)
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item() * imgs.size(0)
        preds = probs.argmax(1)
        correct += (preds == labels).sum().item(); n += imgs.size(0)
        all_scores += probs[:, 1].detach().cpu().tolist()
        all_labels += labels.cpu().tolist()
    torch.set_grad_enabled(True)
    return tot_loss / n, correct / n, all_scores, all_labels


def evaluate_split(scores, labels):
    s = np.array(scores); l = np.array(labels)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (l == 1)).sum()); tn = int(((pred == 0) & (l == 0)).sum())
    fp = int(((pred == 1) & (l == 0)).sum()); fn = int(((pred == 0) & (l == 1)).sum())
    acc = (tp + tn) / len(l)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
    # ROC + AUC
    ths = np.linspace(0, 1, 50)
    roc = []
    P = (l == 1).sum(); N = (l == 0).sum()
    for t in ths[::-1]:
        pr = s >= t
        tpr = int((pr & (l == 1)).sum()) / P if P else 0
        fpr = int((pr & (l == 0)).sum()) / N if N else 0
        roc.append({"fpr": round(float(fpr), 4), "tpr": round(float(tpr), 4), "threshold": round(float(t), 3)})
    pts = sorted([(r["fpr"], r["tpr"]) for r in roc])
    auc = sum((pts[i][0] - pts[i-1][0]) * (pts[i][1] + pts[i-1][1]) / 2 for i in range(1, len(pts)))
    return dict(accuracy=round(acc,4), sensitivity=round(sens,4), specificity=round(spec,4),
                precision=round(prec,4), f1=round(f1,4), auc=round(float(auc),4),
                confusion_matrix=dict(tp=tp, tn=tn, fp=fp, fn=fn), roc_curve=roc)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "busi_data/Dataset_BUSI_with_GT")
    device = torch.device("cpu")
    samples = collect(root)
    tr, va, te = split(samples)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")

    n_benign = sum(1 for _, l in tr if l == 0)
    n_malig = sum(1 for _, l in tr if l == 1)
    weights = torch.tensor([1.0, n_benign / max(1, n_malig)], dtype=torch.float32)
    print(f"Class weights (benign,malignant): {weights.tolist()}")

    tl = DataLoader(BUSIDataset(tr, train=True), batch_size=32, shuffle=True, num_workers=0)
    vl = DataLoader(BUSIDataset(va), batch_size=32, num_workers=0)
    tel = DataLoader(BUSIDataset(te), batch_size=32, num_workers=0)

    model = BreastClassifier().to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    best_auc, best_state = 0.0, None
    EPOCHS = 25
    for ep in range(EPOCHS):
        trl, tra, _, _ = run_epoch(model, tl, criterion, optimizer, device, True)
        vll, vla, vs, vlb = run_epoch(model, vl, criterion, optimizer, device, False)
        scheduler.step()
        vauc = evaluate_split(vs, vlb)["auc"]
        mark = ""
        if vauc > best_auc:
            best_auc = vauc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; mark = " *BEST"
        print(f"Epoch {ep+1:2d}/{EPOCHS}  train_loss={trl:.3f} acc={tra:.3f} | val_loss={vll:.3f} acc={vla:.3f} AUC={vauc:.3f}{mark}")

    if best_state:
        model.load_state_dict(best_state)

    # TEST baholash (ajratilgan to'plam — halol metrikalar)
    _, _, ts, tlb = run_epoch(model, tel, criterion, optimizer, device, False)
    m = evaluate_split(ts, tlb)
    print("\n=== TEST NATIJALARI (ajratilgan to'plam) ===")
    for k in ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]:
        print(f"  {k:12s} {m[k]}")

    # Checkpoint saqlash (xavfsizlik — eksport muvaffaqiyatsiz bo'lsa yo'qolmaydi)
    torch.save(model.state_dict(), "breast_ai_classifier.pth")

    # ONNX eksport (eski interfeysga to'liq mos)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, "breast_ai_model.onnx",
        export_params=True, opset_version=18,
        input_names=["image"], output_names=["class_probs", "birads_score"],
        verbose=False, dynamo=False,
    )
    print("\nbreast_ai_model.onnx yangilandi (haqiqiy model)")

    metrics = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "dataset": "BUSI (Al-Dhabyani 2020) — held-out test split",
        "model": "MobileNetV3-Small, ImageNet pretrained, fine-tuned",
        "split": "stratified 70/15/15, seed=42, test held-out",
        "n_benign": sum(1 for _, l in te if l == 0),
        "n_malignant": sum(1 for _, l in te if l == 1),
        "n_total": len(te),
        **m,
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("metrics.json yozildi (TEST to'plami — data leakage yo'q)")


if __name__ == "__main__":
    main()
