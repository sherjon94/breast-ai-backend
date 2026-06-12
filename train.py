"""
Breast AI — Yagona o'qitish pipeline'i
Konfiguratsiyalanadigan: arxitektura, klass soni, focal loss, augmentatsiya,
k-fold cross-validation (95% CI), test-time augmentation (TTA), ONNX eksport.

Ishlatish:
    # Yakuniy modelni o'qitish + ONNX + metrics.json (single split):
    python train.py --arch efficientnet_b0 --classes 3 --epochs 30 --export

    # K-fold cross-validation (95% CI):
    python train.py --arch efficientnet_b0 --classes 3 --folds 5 --epochs 20

    # Arxitektura taqqoslovi uchun (bitta arxitektura, natijani comparison'ga qo'shadi):
    python train.py --arch resnet18 --classes 3 --epochs 15 --compare

Dataset: busi_data/Dataset_BUSI_with_GT/{normal,benign,malignant}/
"""
import sys, json, argparse, random, math
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

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
DATA_ROOT = Path("busi_data/Dataset_BUSI_with_GT")
# malignant DOIM oxirgi indeks (binar saraton-aniqlash metrikalari uchun)
CLASSES_3 = ["normal", "benign", "malignant"]
CLASSES_2 = ["benign", "malignant"]


# ─── Dataset ──────────────────────────────────────────────────────────────────

def collect(classes):
    samples = []
    for idx, c in enumerate(classes):
        for p in (DATA_ROOT / c).glob("*.png"):
            if "mask" in p.stem.lower():
                continue
            samples.append((str(p), idx))
    random.shuffle(samples)
    return samples


class BUSI(Dataset):
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), label


# ─── Model ────────────────────────────────────────────────────────────────────

def build_backbone(arch, num_classes):
    if arch == "mobilenet_v3":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Noma'lum arxitektura: {arch}")
    return m


class Net(nn.Module):
    """Backbone -> softmax class_probs + birads_score (malignant ehtimoli)"""
    def __init__(self, arch, num_classes):
        super().__init__()
        self.backbone = build_backbone(arch, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        probs = F.softmax(self.backbone(x), dim=1)
        birads = probs[:, -1:]  # malignant DOIM oxirgi klass
        return probs, birads


class FocalLoss(nn.Module):
    """Class imbalance uchun focal loss"""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight; self.gamma = gamma

    def forward(self, logp, target):
        p = torch.exp(logp)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        focal = (1 - p.gather(1, target[:, None]).squeeze(1)) ** self.gamma
        return (focal * ce).mean()


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def run(model, loader, criterion, opt, device, train, tta=False):
    model.train() if train else model.eval()
    tot, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        probs, _ = model(imgs)
        if tta and not train:  # horizontal flip TTA
            probs2, _ = model(torch.flip(imgs, dims=[3]))
            probs = (probs + probs2) / 2
        loss = criterion(torch.log(probs + 1e-9), labels)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * imgs.size(0)
        correct += (probs.argmax(1) == labels).sum().item(); n += imgs.size(0)
        all_probs += probs.detach().cpu().tolist()
        all_labels += labels.cpu().tolist()
    torch.set_grad_enabled(True)
    return tot / n, correct / n, np.array(all_probs), np.array(all_labels)


def binary_cancer_metrics(probs, labels, malignant_idx):
    """Saraton-aniqlash (malignant vs qolganlar) binar metrikalari"""
    score = probs[:, malignant_idx]
    y = (labels == malignant_idx).astype(int)
    pred = (score >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
    # ROC + AUC
    ths = np.linspace(0, 1, 50); roc = []
    P, N = (y == 1).sum(), (y == 0).sum()
    for t in ths[::-1]:
        pr = score >= t
        roc.append({"fpr": round(float((pr & (y == 0)).sum() / N) if N else 0, 4),
                    "tpr": round(float((pr & (y == 1)).sum() / P) if P else 0, 4),
                    "threshold": round(float(t), 3)})
    pts = sorted([(r["fpr"], r["tpr"]) for r in roc])
    auc = sum((pts[i][0] - pts[i-1][0]) * (pts[i][1] + pts[i-1][1]) / 2 for i in range(1, len(pts)))
    return dict(sensitivity=round(sens, 4), specificity=round(spec, 4), precision=round(prec, 4),
                f1=round(f1, 4), auc=round(float(auc), 4),
                confusion_matrix=dict(tp=tp, tn=tn, fp=fp, fn=fn), roc_curve=roc)


def confusion_nxn(probs, labels, n):
    cm = np.zeros((n, n), dtype=int)
    pred = probs.argmax(1)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    return cm.tolist()


def stratified_folds(samples, k):
    by_cls = {}
    for s in samples:
        by_cls.setdefault(s[1], []).append(s)
    folds = [[] for _ in range(k)]
    for items in by_cls.values():
        for i, s in enumerate(items):
            folds[i % k].append(s)
    return folds


def train_one(samples_tr, samples_va, arch, num_classes, epochs, device, use_focal, tta):
    n_per = [sum(1 for _, l in samples_tr if l == c) for c in range(num_classes)]
    total = sum(n_per)
    weights = torch.tensor([total / (num_classes * max(1, n)) for n in n_per], dtype=torch.float32).to(device)
    tl = DataLoader(BUSI(samples_tr, train=True), batch_size=32, shuffle=True)
    vl = DataLoader(BUSI(samples_va), batch_size=32)
    model = Net(arch, num_classes).to(device)
    criterion = FocalLoss(weight=weights) if use_focal else nn.NLLLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_auc, best_state = -1, None
    for ep in range(epochs):
        trl, tra, _, _ = run(model, tl, criterion, opt, device, True)
        _, vacc, vp, vlb = run(model, vl, criterion, opt, device, False, tta=tta)
        sched.step()
        vauc = binary_cancer_metrics(vp, vlb, num_classes - 1)["auc"]
        if vauc > best_auc:
            best_auc = vauc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep+1:2d}/{epochs} train_loss={trl:.3f} acc={tra:.3f} | val_acc={vacc:.3f} cancer_AUC={vauc:.3f}")
    model.load_state_dict(best_state)
    return model


def ci95(values):
    n = len(values)
    if n < 2:
        return round(float(values[0]), 4), 0.0
    m = float(np.mean(values)); sd = float(np.std(values, ddof=1))
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 1.96)
    return round(m, 4), round(t * sd / math.sqrt(n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0", choices=["mobilenet_v3", "resnet18", "efficientnet_b0"])
    ap.add_argument("--classes", type=int, default=3, choices=[2, 3])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=1, help=">1 = k-fold CV")
    ap.add_argument("--focal", action="store_true", default=True)
    ap.add_argument("--no-focal", dest="focal", action="store_false")
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--export", action="store_true", help="ONNX + metrics.json yozish")
    ap.add_argument("--compare", action="store_true", help="comparison.json ga qo'shish")
    args = ap.parse_args()

    device = torch.device("cpu")
    classes = CLASSES_3 if args.classes == 3 else CLASSES_2
    samples = collect(classes)
    mal_idx = args.classes - 1
    print(f"Arch={args.arch} classes={classes} focal={args.focal} tta={args.tta} folds={args.folds}")
    print(f"Jami {len(samples)} rasm: " + ", ".join(f"{c}={sum(1 for _,l in samples if l==i)}" for i, c in enumerate(classes)))

    # ── K-FOLD CV ──
    if args.folds > 1:
        folds = stratified_folds(samples, args.folds)
        agg = {"accuracy": [], "sensitivity": [], "specificity": [], "auc": [], "f1": []}
        for fi in range(args.folds):
            print(f"\n=== Fold {fi+1}/{args.folds} ===")
            va = folds[fi]; tr = [s for j, f in enumerate(folds) if j != fi for s in f]
            model = train_one(tr, va, args.arch, args.classes, args.epochs, device, args.focal, args.tta)
            _, vacc, vp, vlb = run(model, DataLoader(BUSI(va), batch_size=32), nn.NLLLoss(), None, device, False, tta=args.tta)
            bm = binary_cancer_metrics(vp, vlb, mal_idx)
            agg["accuracy"].append(vacc); agg["sensitivity"].append(bm["sensitivity"])
            agg["specificity"].append(bm["specificity"]); agg["auc"].append(bm["auc"]); agg["f1"].append(bm["f1"])
            print(f"  Fold {fi+1}: acc={vacc:.3f} sens={bm['sensitivity']:.3f} spec={bm['specificity']:.3f} AUC={bm['auc']:.3f}")
        cv = {}
        for k, vals in agg.items():
            m, ci = ci95(vals)
            cv[k] = {"mean": m, "ci95": ci, "folds": [round(float(v), 4) for v in vals]}
        out = {"arch": args.arch, "classes": args.classes, "folds": args.folds,
               "evaluated_at": datetime.utcnow().isoformat(), "cross_validation": cv}
        Path("cv_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n=== K-FOLD NATIJALARI (mean ± 95% CI) ===")
        for k, v in cv.items():
            print(f"  {k:12s} {v['mean']:.3f} ± {v['ci95']:.3f}")
        print("cv_metrics.json yozildi")
        return

    # ── SINGLE SPLIT (export / compare) ──
    folds = stratified_folds(samples, 100)  # ~1% birlik
    # stratified 70/15/15
    by_cls = {}
    for s in samples:
        by_cls.setdefault(s[1], []).append(s)
    tr, va, te = [], [], []
    for items in by_cls.values():
        n = len(items); a, b = int(n*0.70), int(n*0.85)
        tr += items[:a]; va += items[a:b]; te += items[b:]
    random.shuffle(tr)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")
    model = train_one(tr, va, args.arch, args.classes, args.epochs, device, args.focal, args.tta)

    _, tacc, tp, tlb = run(model, DataLoader(BUSI(te), batch_size=32), nn.NLLLoss(), None, device, False, tta=args.tta)
    bm = binary_cancer_metrics(tp, tlb, mal_idx)
    cm_n = confusion_nxn(tp, tlb, args.classes)
    print(f"\n=== TEST (ajratilgan) === arch={args.arch}")
    print(f"  {args.classes}-klass accuracy: {tacc:.4f}")
    print(f"  cancer AUC={bm['auc']} sens={bm['sensitivity']} spec={bm['specificity']} f1={bm['f1']}")

    if args.compare:
        comp_path = Path("comparison.json")
        data = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {"results": []}
        data["results"] = [r for r in data["results"] if r["arch"] != args.arch]
        data["results"].append({"arch": args.arch, "accuracy": round(tacc, 4),
                                 "auc": bm["auc"], "sensitivity": bm["sensitivity"],
                                 "specificity": bm["specificity"], "f1": bm["f1"]})
        data["evaluated_at"] = datetime.utcnow().isoformat()
        comp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"comparison.json yangilandi ({args.arch})")

    if args.export:
        torch.save(model.state_dict(), "breast_ai_classifier.pth")
        model.eval()
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), "breast_ai_model.onnx",
                          export_params=True, opset_version=18,
                          input_names=["image"], output_names=["class_probs", "birads_score"],
                          verbose=False, dynamo=False)
        metrics = {
            "evaluated_at": datetime.utcnow().isoformat(),
            "dataset": "BUSI (Al-Dhabyani 2020) — held-out test split",
            "model": f"{args.arch}, ImageNet pretrained, fine-tuned ({args.classes}-class)",
            "classes": classes,
            "split": "stratified 70/15/15, seed=42, test held-out",
            "n_total": len(te),
            "n_per_class": {c: int((tlb == i).sum()) for i, c in enumerate(classes)},
            "multiclass_accuracy": round(tacc, 4),
            "confusion_matrix_nxn": cm_n,
            "cancer_detection": {  # malignant vs qolganlar (klinik asosiy metrika)
                "accuracy": round(tacc, 4) if args.classes == 2 else None,
                "sensitivity": bm["sensitivity"], "specificity": bm["specificity"],
                "precision": bm["precision"], "f1": bm["f1"], "auc": bm["auc"],
            },
            # Frontend mosligi uchun yuqori darajadagi maydonlar:
            "accuracy": round(tacc, 4), "sensitivity": bm["sensitivity"],
            "specificity": bm["specificity"], "precision": bm["precision"],
            "f1": bm["f1"], "auc": bm["auc"],
            "confusion_matrix": bm["confusion_matrix"], "roc_curve": bm["roc_curve"],
        }
        # CV natijalari bo'lsa qo'shish
        cvp = Path("cv_metrics.json")
        if cvp.exists():
            try:
                metrics["cross_validation"] = json.loads(cvp.read_text(encoding="utf-8")).get("cross_validation")
            except Exception:
                pass
        compp = Path("comparison.json")
        if compp.exists():
            try:
                metrics["architecture_comparison"] = json.loads(compp.read_text(encoding="utf-8")).get("results")
            except Exception:
                pass
        Path("metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print("breast_ai_model.onnx + metrics.json yozildi")


if __name__ == "__main__":
    main()
