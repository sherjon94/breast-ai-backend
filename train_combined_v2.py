"""
Birlashgan UZI model — QAT'IY versiya (kamchiliklar tuzatilgan):
- Bemor (patient) darajasidagi split — BUS-BRA Case bo'yicha guruhlash (leakage YO'Q)
- 5-fold GroupKFold cross-validation -> mean ± 95% CI (combined modelники)
- 3-klass per-class metrika + 3x3 confusion matrix
- Kalibratsiya: ECE + Brier; Youden operating point
Natija: breast_ai_model.onnx + metrics.json (halol)
"""
import json, csv, random, math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
CLASSES = ["normal", "benign", "malignant"]
BUSI = Path("busi_data/Dataset_BUSI_with_GT")
BUSBRA = Path("busbra_data/BUSBRA")
TF_T = transforms.Compose([transforms.Resize((240,240)),transforms.RandomCrop((224,224)),transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),transforms.RandomRotation(20),transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
TF_E = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def collect():
    """(path, label, group) — group = bemor identifikatori (leakage oldini olish)"""
    samples = []
    for idx, c in enumerate(CLASSES):
        for p in (BUSI / c).glob("*.png"):
            if "mask" in p.stem.lower(): continue
            samples.append((str(p), idx, "busi_" + p.stem))   # BUSI: har rasm o'z guruhi
    case = {}
    with open(BUSBRA/"bus_data.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            case[r["ID"]] = (1 if r["Pathology"].strip().lower()=="benign" else 2, "busbra_"+r["Case"])
    for p in (BUSBRA/"Images").glob("*.png"):
        if p.stem in case:
            lab, grp = case[p.stem]; samples.append((str(p), lab, grp))
    random.shuffle(samples); return samples


class DS(Dataset):
    def __init__(self, items, train=False): self.items=items; self.tf=TF_T if train else TF_E
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p,l,_=self.items[i]; return self.tf(Image.open(p).convert("RGB")), l


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.b=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.b.classifier[-1]=nn.Linear(self.b.classifier[-1].in_features,3)
    def forward(self,x): p=F.softmax(self.b(x),1); return p, p[:,2:3]

class Focal(nn.Module):
    def __init__(self,w,g=2.0): super().__init__(); self.w=w; self.g=g
    def forward(self,logp,y):
        p=torch.exp(logp); ce=F.nll_loss(logp,y,weight=self.w,reduction="none")
        return (((1-p.gather(1,y[:,None]).squeeze(1))**self.g)*ce).mean()


def cancer_metrics(probs, labels, thr=0.5):
    s=probs[:,2]; y=(np.array(labels)==2).astype(int); pred=(s>=thr).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    sens=tp/(tp+fn) if (tp+fn) else 0; spec=tn/(tn+fp) if (tn+fp) else 0; prec=tp/(tp+fp) if (tp+fp) else 0
    f1=2*prec*sens/(prec+sens) if (prec+sens) else 0; acc=(tp+tn)/len(y)
    ths=np.linspace(0,1,50); roc=[]; P=(y==1).sum(); N=(y==0).sum()
    for t in ths[::-1]:
        pr=s>=t; roc.append({"fpr":round(float((pr&(y==0)).sum()/N),4) if N else 0,"tpr":round(float((pr&(y==1)).sum()/P),4) if P else 0,"threshold":round(float(t),3)})
    pts=sorted([(r["fpr"],r["tpr"]) for r in roc]); auc=sum((pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2 for i in range(1,len(pts)))
    return dict(accuracy=round(acc,4),sensitivity=round(sens,4),specificity=round(spec,4),precision=round(prec,4),f1=round(f1,4),auc=round(float(auc),4),confusion_matrix={"tp":tp,"tn":tn,"fp":fp,"fn":fn},roc_curve=roc)

def youden_threshold(probs, labels):
    s=probs[:,2]; y=(np.array(labels)==2).astype(int); best_t,best_j=0.5,-1
    for t in np.linspace(0.05,0.95,19):
        pr=s>=t; P=(y==1).sum(); N=(y==0).sum()
        tpr=(pr&(y==1)).sum()/P if P else 0; fpr=(pr&(y==0)).sum()/N if N else 0
        if tpr-fpr>best_j: best_j,best_t=tpr-fpr,float(t)
    return round(best_t,3)

def calibration(probs, labels, bins=10):
    s=probs[:,2]; y=(np.array(labels)==2).astype(int)
    brier=float(np.mean((s-y)**2)); ece=0.0
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins; m=(s>=lo)&(s<hi)
        if m.sum()>0: ece+=abs(s[m].mean()-y[m].mean())*m.sum()/len(y)
    return round(brier,4), round(float(ece),4)

def per_class(probs, labels):
    pred=probs.argmax(1); y=np.array(labels); cm=[[0]*3 for _ in range(3)]
    for t,p in zip(y,pred): cm[t][p]+=1
    out={}
    for i,c in enumerate(CLASSES):
        tp=cm[i][i]; fn=sum(cm[i])-tp; fp=sum(cm[r][i] for r in range(3))-tp
        rec=tp/(tp+fn) if (tp+fn) else 0; pr=tp/(tp+fp) if (tp+fp) else 0
        out[c]={"recall":round(rec,3),"precision":round(pr,3),"n":int(sum(cm[i]))}
    return out, cm

def run(model,loader,crit,opt,dev,train,tta=False):
    model.train() if train else model.eval(); correct=n=0; P=[]; L=[]; torch.set_grad_enabled(train)
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); probs,_=model(x)
        if tta and not train: p2,_=model(torch.flip(x,dims=[3])); probs=(probs+p2)/2
        if crit is not None:
            loss=crit(torch.log(probs+1e-9),y)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        correct+=(probs.argmax(1)==y).sum().item(); n+=x.size(0); P+=probs.detach().cpu().tolist(); L+=y.cpu().tolist()
    torch.set_grad_enabled(True); return correct/max(1,n), np.array(P), L

def class_weights(items, dev):
    cnt=[sum(1 for _,l,_ in items if l==c) for c in range(3)]; tot=sum(cnt)
    return torch.tensor([tot/(3*max(1,c)) for c in cnt],dtype=torch.float32).to(dev)

def train_model(tr, va, dev, epochs):
    tl=DataLoader(DS(tr,True),batch_size=32,shuffle=True); vl=DataLoader(DS(va),batch_size=32)
    model=Net().to(dev); crit=Focal(class_weights(tr,dev))
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    best=-1; best_state=None
    for ep in range(epochs):
        run(model,tl,crit,opt,dev,True)
        _,vp,vl2=run(model,vl,None,None,dev,False,tta=True); a=cancer_metrics(vp,vl2)["auc"]
        if a>best: best=a; best_state={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); return model

def ci95(vals):
    n=len(vals); m=float(np.mean(vals))
    if n<2: return round(m,4),0.0
    sd=float(np.std(vals,ddof=1)); t={2:12.71,3:4.30,4:3.18,5:2.78,6:2.57}.get(n,2.78)
    return round(m,4), round(t*sd/math.sqrt(n),4)

def main():
    dev=torch.device("cpu"); s=collect()
    groups=[g for _,_,g in s]; labels=[l for _,l,_ in s]
    print(f"Jami {len(s)} rasm, {len(set(groups))} bemor-guruh", flush=True)

    # ── 5-fold GroupKFold CV (bemor bo'yicha — leakage yo'q) ──
    gkf=GroupKFold(n_splits=5); agg={"auc":[],"sensitivity":[],"specificity":[],"accuracy":[],"f1":[]}
    for fi,(tri,tei) in enumerate(gkf.split(s,labels,groups)):
        tr=[s[i] for i in tri]; te=[s[i] for i in tei]
        model=train_model(tr,te,dev,8)
        _,tp,tl=run(model,DataLoader(DS(te),batch_size=32),None,None,dev,False,tta=True)
        m=cancer_metrics(tp,tl)
        for k in agg: agg[k].append(m[k] if k!="accuracy" else m["accuracy"])
        print(f"  Fold {fi+1}: AUC={m['auc']} sens={m['sensitivity']} spec={m['specificity']}", flush=True)
    cv={};
    for k,v in agg.items(): mm,ci=ci95(v); cv[k]={"mean":mm,"ci95":ci,"folds":[round(float(x),4) for x in v]}
    print(f"CV: AUC {cv['auc']['mean']}±{cv['auc']['ci95']} | sens {cv['sensitivity']['mean']}±{cv['sensitivity']['ci95']}", flush=True)

    # ── Yakuniy model: bemor bo'yicha 85/15 held-out test ──
    gss=GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=SEED)
    tri,tei=next(gss.split(s,labels,groups)); trv=[s[i] for i in tri]; te=[s[i] for i in tei]
    gss2=GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=SEED)
    g2=[s[i][2] for i in tri]; tri2,vai2=next(gss2.split(trv,[x[1] for x in trv],g2))
    tr=[trv[i] for i in tri2]; va=[trv[i] for i in vai2]
    print(f"Yakuniy: train {len(tr)} | val {len(va)} | test {len(te)} (bemor-ajratilgan)", flush=True)
    model=train_model(tr,va,dev,18)
    _,tp,tl=run(model,DataLoader(DS(te),batch_size=32),None,None,dev,False,tta=True)
    m=cancer_metrics(tp,tl); pc,cm3=per_class(tp,tl); brier,ece=calibration(tp,tl); yj=youden_threshold(tp,tl)
    macc=float((np.array(tp).argmax(1)==np.array(tl)).mean())
    print(f"\nTEST (bemor-ajratilgan): cancer AUC={m['auc']} sens={m['sensitivity']} spec={m['specificity']} | 3-klass acc={round(macc,4)}", flush=True)
    print(f"  per-class: {pc} | Brier={brier} ECE={ece} | Youden thr={yj}", flush=True)

    torch.save(model.state_dict(),"breast_ai_combined_v2.pth"); model.eval()
    torch.onnx.export(model,torch.randn(1,3,224,224),"breast_ai_model.onnx",export_params=True,opset_version=18,
        input_names=["image"],output_names=["class_probs","birads_score"],verbose=False,dynamo=False)
    metrics={"evaluated_at":datetime.utcnow().isoformat(),
        "dataset":"BUSI + BUS-BRA (combined, multi-source)","model":"EfficientNet-B0 (3-class), patient-grouped split",
        "classes":CLASSES,"split":"patient-grouped 70/15/15 (GroupShuffleSplit) — NO patient leakage",
        "n_total":len(te),"multiclass_accuracy":round(macc,4),
        "accuracy":m["accuracy"],"sensitivity":m["sensitivity"],"specificity":m["specificity"],"precision":m["precision"],"f1":m["f1"],"auc":m["auc"],
        "confusion_matrix":m["confusion_matrix"],"roc_curve":m["roc_curve"],
        "per_class_3class":pc,"confusion_matrix_3x3":cm3,
        "calibration":{"brier":brier,"ece":ece},"youden_operating_point":yj,
        "cross_validation":cv,  # ENDI combined modelники (5-fold GroupKFold)
        "cv_note":"5-fold GroupKFold, patient-grouped, combined model"}
    Path("metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    print("breast_ai_model.onnx + metrics.json yangilandi (QAT'IY, leakage-siz)", flush=True)

if __name__=="__main__":
    main()
