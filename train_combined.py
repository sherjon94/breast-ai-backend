"""
Birlashgan model — BUSI (Misr) + BUS-BRA (Braziliya) datasetlarida birga o'qitish.
Domain shift muammosini yengib, mustahkamroq (generalizatsiyalanuvchi) model beradi.

3-klass: normal / benign / malignant (normal faqat BUSI'da bor).
Stratifikatsiyalangan 70/15/15 split — test to'plamida IKKALA manba ham bor.

Ishlatish:  python train_combined.py
Natija:     breast_ai_model.onnx + metrics.json
"""
import sys, json, csv, random, math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
CLASSES = ["normal", "benign", "malignant"]  # malignant = oxirgi
BUSI = Path("busi_data/Dataset_BUSI_with_GT")
BUSBRA = Path("busbra_data/BUSBRA")


def collect():
    samples = []  # (path, label, source)
    # BUSI — papka tuzilishi (3 klass)
    for idx, c in enumerate(CLASSES):
        for p in (BUSI / c).glob("*.png"):
            if "mask" not in p.stem.lower():
                samples.append((str(p), idx, "BUSI"))
    # BUS-BRA — CSV (benign/malignant)
    labmap = {}
    with open(BUSBRA / "bus_data.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            labmap[r["ID"]] = 1 if r["Pathology"].strip().lower() == "benign" else 2  # benign=1, malignant=2
    for p in (BUSBRA / "Images").glob("*.png"):
        if p.stem in labmap:
            samples.append((str(p), labmap[p.stem], "BUS-BRA"))
    random.shuffle(samples)
    return samples


class DS(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples
        self.tf = transforms.Compose(([
            transforms.Resize((240, 240)), transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ] if train else [transforms.Resize((224, 224))]) + [
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.b.classifier[-1] = nn.Linear(self.b.classifier[-1].in_features, 3)
    def forward(self, x):
        p = F.softmax(self.b(x), 1)
        return p, p[:, 2:3]


class Focal(nn.Module):
    def __init__(self, w, g=2.0): super().__init__(); self.w = w; self.g = g
    def forward(self, logp, y):
        p = torch.exp(logp)
        ce = F.nll_loss(logp, y, weight=self.w, reduction="none")
        return (((1 - p.gather(1, y[:, None]).squeeze(1)) ** self.g) * ce).mean()


def split(samples):
    by = {}
    for s in samples: by.setdefault(s[1], []).append(s)
    tr, va, te = [], [], []
    for items in by.values():
        n = len(items); a, b = int(n*0.70), int(n*0.85)
        tr += items[:a]; va += items[a:b]; te += items[b:]
    random.shuffle(tr)
    return tr, va, te


def cancer_metrics(probs, labels):
    s = probs[:, 2]; y = (labels == 2).astype(int)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    sens = tp/(tp+fn) if (tp+fn) else 0; spec = tn/(tn+fp) if (tn+fp) else 0
    prec = tp/(tp+fp) if (tp+fp) else 0; f1 = 2*prec*sens/(prec+sens) if (prec+sens) else 0
    ths = np.linspace(0,1,50); roc=[]; P=(y==1).sum(); N=(y==0).sum()
    for t in ths[::-1]:
        pr=s>=t
        roc.append({"fpr":round(float((pr&(y==0)).sum()/N),4),"tpr":round(float((pr&(y==1)).sum()/P),4),"threshold":round(float(t),3)})
    pts=sorted([(r["fpr"],r["tpr"]) for r in roc])
    auc=sum((pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2 for i in range(1,len(pts)))
    return dict(sensitivity=round(sens,4),specificity=round(spec,4),precision=round(prec,4),
                f1=round(f1,4),auc=round(float(auc),4),confusion_matrix={"tp":tp,"tn":tn,"fp":fp,"fn":fn},roc_curve=roc)


def run(model, loader, crit, opt, dev, train, tta=False):
    model.train() if train else model.eval()
    correct=n=0; P=[]; L=[]
    torch.set_grad_enabled(train)
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        probs,_=model(x)
        if tta and not train:
            p2,_=model(torch.flip(x,dims=[3])); probs=(probs+p2)/2
        loss=crit(torch.log(probs+1e-9),y)
        if train: opt.zero_grad(); loss.backward(); opt.step()
        correct+=(probs.argmax(1)==y).sum().item(); n+=x.size(0)
        P+=probs.detach().cpu().tolist(); L+=y.cpu().tolist()
    torch.set_grad_enabled(True)
    return correct/n, np.array(P), np.array(L)


def main():
    dev=torch.device("cpu")
    samples=collect()
    src={}
    for _,_,s in samples: src[s]=src.get(s,0)+1
    cls=[0,0,0]
    for _,l,_ in samples: cls[l]+=1
    print(f"Jami {len(samples)}: normal={cls[0]} benign={cls[1]} malignant={cls[2]} | manba={src}")
    tr,va,te=split(samples)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")

    w=torch.tensor([len(tr)/(3*max(1,sum(1 for _,l,_ in tr if l==c))) for c in range(3)],dtype=torch.float32).to(dev)
    tl=DataLoader(DS(tr,True),batch_size=32,shuffle=True)
    vl=DataLoader(DS(va),batch_size=32); tel=DataLoader(DS(te),batch_size=32)
    model=Net().to(dev); crit=Focal(w)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=1e-6)
    best=-1; best_state=None
    for ep in range(20):
        tra,_,_=run(model,tl,crit,opt,dev,True)
        vacc,vp,vl2=run(model,vl,crit,opt,dev,False,tta=True)
        sch.step()
        vauc=cancer_metrics(vp,vl2)["auc"]
        if vauc>best: best=vauc; best_state={k:v.clone() for k,v in model.state_dict().items()}
        print(f"ep {ep+1:2d}/20 train_acc={tra:.3f} val_acc={vacc:.3f} cancer_AUC={vauc:.3f}{' *' if vauc==best else ''}",flush=True)
    model.load_state_dict(best_state)

    tacc,tp,tl2=run(model,tel,crit,opt,dev,False,tta=True)
    m=cancer_metrics(tp,tl2)
    print(f"\n=== TEST (birlashgan, ikkala manba) ===\n  3-klass acc={tacc:.4f} cancer AUC={m['auc']} sens={m['sensitivity']} spec={m['specificity']}")

    torch.save(model.state_dict(),"breast_ai_combined.pth")
    model.eval()
    torch.onnx.export(model,torch.randn(1,3,224,224),"breast_ai_model.onnx",export_params=True,
        opset_version=18,input_names=["image"],output_names=["class_probs","birads_score"],verbose=False,dynamo=False)

    metrics={
        "evaluated_at":datetime.utcnow().isoformat(),
        "dataset":"BUSI (Egypt) + BUS-BRA (Brazil) — combined, multi-source",
        "model":"EfficientNet-B0, ImageNet pretrained, combined-dataset fine-tuned (3-class)",
        "classes":CLASSES,"split":"stratified 70/15/15, seed=42, test held-out (both sources)",
        "n_total":len(te),"n_train":len(tr),
        "sources":src,
        "multiclass_accuracy":round(tacc,4),
        "accuracy":round(tacc,4),"sensitivity":m["sensitivity"],"specificity":m["specificity"],
        "precision":m["precision"],"f1":m["f1"],"auc":m["auc"],
        "confusion_matrix":m["confusion_matrix"],"roc_curve":m["roc_curve"],
        "cancer_detection":{"sensitivity":m["sensitivity"],"specificity":m["specificity"],"auc":m["auc"]},
    }
    # avvalgi CV/comparison/external bo'lsa qo'shish
    for fn,key in [("cv_metrics.json","cross_validation"),("external_metrics.json","external_validation_busi_only")]:
        p=Path(fn)
        if p.exists():
            try:
                d=json.loads(p.read_text(encoding="utf-8"))
                metrics[key]=d.get("cross_validation",d)
            except Exception: pass
    cp=Path("comparison.json")
    if cp.exists():
        try: metrics["architecture_comparison"]=json.loads(cp.read_text(encoding="utf-8")).get("results")
        except Exception: pass
    Path("metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    print("breast_ai_model.onnx + metrics.json yangilandi (birlashgan model)")


if __name__=="__main__":
    main()
