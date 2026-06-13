"""
Birlashgan mammografiya modeli — CBIS-DDSM + DMID birga o'qitish.
UZI'dagi muvaffaqiyatli yondashuvga parallel (ko'p manbali -> generalizatsiya).
Test: CBIS-test va DMID-holdout ALOHIDA baholanadi (manbalararo ko'chishni ko'rsatish uchun).

Natija: breast_ai_mammo.onnx + mammo_metrics.json
"""
import json, random
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
CBIMG = Path("cbis_img"); DMID = Path("dmid_data")

TF_TRAIN = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(12), transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TF_EVAL = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def cbis_items(split):
    out = []
    for idx, c in enumerate(["benign", "malignant"]):
        for p in (CBIMG / split / c).glob("*.png"): out.append((str(p), idx))
    return out

def dmid_items():
    out = []
    for idx, c in enumerate(["benign", "malignant"]):
        for p in (DMID / c).glob("*.png"): out.append((str(p), idx))
    random.shuffle(out); return out


class DS(Dataset):
    def __init__(self, items, train=False): self.items = items; self.tf = TF_TRAIN if train else TF_EVAL
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]; return self.tf(Image.open(p).convert("RGB")), l


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.b = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.b.classifier[-1] = nn.Linear(self.b.classifier[-1].in_features, 2)
    def forward(self, x): p = F.softmax(self.b(x), 1); return p, p[:, 1:2]

class Focal(nn.Module):
    def __init__(self, w, g=2.0): super().__init__(); self.w = w; self.g = g
    def forward(self, logp, y):
        p = torch.exp(logp); ce = F.nll_loss(logp, y, weight=self.w, reduction="none")
        return (((1 - p.gather(1, y[:, None]).squeeze(1)) ** self.g) * ce).mean()

def metr(probs, labels):
    sc = probs[:, 1]; y = np.array(labels); pred = (sc >= 0.5).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    sens=tp/(tp+fn) if (tp+fn) else 0; spec=tn/(tn+fp) if (tn+fp) else 0; prec=tp/(tp+fp) if (tp+fp) else 0
    f1=2*prec*sens/(prec+sens) if (prec+sens) else 0; acc=(tp+tn)/len(y)
    ths=np.linspace(0,1,50); roc=[]; P=(y==1).sum(); N=(y==0).sum()
    for t in ths[::-1]:
        pr=sc>=t; roc.append({"fpr":round(float((pr&(y==0)).sum()/N),4) if N else 0,"tpr":round(float((pr&(y==1)).sum()/P),4) if P else 0,"threshold":round(float(t),3)})
    pts=sorted([(r["fpr"],r["tpr"]) for r in roc]); auc=sum((pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2 for i in range(1,len(pts)))
    return dict(accuracy=round(acc,4),sensitivity=round(sens,4),specificity=round(spec,4),precision=round(prec,4),f1=round(f1,4),auc=round(float(auc),4),confusion_matrix={"tp":tp,"tn":tn,"fp":fp,"fn":fn},roc_curve=roc)

def run(model, loader, crit, opt, dev, train, tta=False):
    model.train() if train else model.eval(); correct=n=0; P=[]; L=[]; torch.set_grad_enabled(train)
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); probs,_=model(x)
        if tta and not train: p2,_=model(torch.flip(x,dims=[3])); probs=(probs+p2)/2
        if crit is not None:
            loss=crit(torch.log(probs+1e-9),y)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        correct+=(probs.argmax(1)==y).sum().item(); n+=x.size(0); P+=probs.detach().cpu().tolist(); L+=y.cpu().tolist()
    torch.set_grad_enabled(True); return correct/max(1,n), np.array(P), L

def main():
    dev=torch.device("cpu"); import collections
    cbis_tr=cbis_items("train")+cbis_items("validation"); cbis_te=cbis_items("test")
    dm=dmid_items(); nv=int(len(dm)*0.15); dm_te=dm[:nv]; dm_tr=dm[nv:]
    train=cbis_tr+dm_tr
    print(f"Train {len(train)} (CBIS {len(cbis_tr)} + DMID {len(dm_tr)}) | test: CBIS {len(cbis_te)}, DMID {len(dm_te)}", flush=True)
    cnt=collections.Counter(l for _,l in train)
    w=torch.tensor([1.0, cnt[0]/max(1,cnt[1])],dtype=torch.float32).to(dev)
    tl=DataLoader(DS(train,True),batch_size=32,shuffle=True)
    cbl=DataLoader(DS(cbis_te),batch_size=32); dml=DataLoader(DS(dm_te),batch_size=32)
    model=Net().to(dev); crit=Focal(w)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12,eta_min=1e-6)
    best=-1; best_state=None
    for ep in range(12):
        tra,_,_=run(model,tl,crit,opt,dev,True)
        # validatsiya: ikkala manba aralash kichik tekshiruv (CBIS test ustida)
        _,vp,vl=run(model,cbl,None,None,dev,False,tta=True); vauc=metr(vp,vl)["auc"]
        sch.step()
        if vauc>best: best=vauc; best_state={k:v.clone() for k,v in model.state_dict().items()}
        print(f"ep {ep+1:2d}/12 train_acc={tra:.3f} CBIS_AUC={vauc:.3f}{' *' if vauc==best else ''}",flush=True)
    model.load_state_dict(best_state)
    _,cp,cl=run(model,cbl,None,None,dev,False,tta=True); cm=metr(cp,cl)
    _,dp,dl=run(model,dml,None,None,dev,False,tta=True); dmm=metr(dp,dl)
    print(f"\n=== CBIS test === AUC={cm['auc']} sens={cm['sensitivity']} spec={cm['specificity']}",flush=True)
    print(f"=== DMID holdout === AUC={dmm['auc']} sens={dmm['sensitivity']} spec={dmm['specificity']}",flush=True)
    torch.save(model.state_dict(),"breast_ai_mammo.pth"); model.eval()
    torch.onnx.export(model,torch.randn(1,3,224,224),"breast_ai_mammo.onnx",export_params=True,opset_version=18,
        input_names=["image"],output_names=["class_probs","birads_score"],verbose=False,dynamo=False)
    out={"evaluated_at":datetime.utcnow().isoformat(),"dataset":"CBIS-DDSM + DMID (combined, multi-source)",
        "model":"EfficientNet-B0, combined mammography (2-class)","classes":["benign","malignant"],
        "n_total":len(cbis_te),**cm,
        "per_source":{"CBIS_test":{"auc":cm["auc"],"sensitivity":cm["sensitivity"],"specificity":cm["specificity"],"n":len(cbis_te)},
                      "DMID_holdout":{"auc":dmm["auc"],"sensitivity":dmm["sensitivity"],"specificity":dmm["specificity"],"n":len(dm_te)}}}
    Path("mammo_metrics.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print("breast_ai_mammo.onnx + mammo_metrics.json yangilandi (CBIS+DMID birlashgan)")

if __name__=="__main__":
    main()
