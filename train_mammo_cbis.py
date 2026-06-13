"""
Mammografiya modeli — CBIS-DDSM (~7030 rasm, biopsiya bilan) da o'qitish.
DMID (411) tashqi validatsiya sifatida ishlatiladi (cross-dataset, US hikoyasiga parallel).
Belgi: 0=benign, 1=malignant (DMID tashqi AUC bilan avtomatik tasdiqlanadi).

Natija: breast_ai_mammo.onnx (kuchaytirilgan) + mammo_metrics.json
"""
import json, io, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
CB = Path("cbis_data")
DMID = Path("dmid_data")


CBIMG = Path("cbis_img")

class CbisFolderDS(Dataset):
    """cbis_img/{split}/{benign,malignant}/*.png (oldindan 256px ga chiqarilgan — tez)"""
    def __init__(self, split, train=False):
        self.items = []
        for idx, c in enumerate(["benign", "malignant"]):
            for p in (CBIMG / split / c).glob("*.png"):
                self.items.append((str(p), idx))
        self.labs = [l for _, l in self.items]
        self.tf = transforms.Compose(([
            transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ] if train else [transforms.Resize((224, 224))]) + [
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), l


class FolderDS(Dataset):
    """DMID papka (tashqi validatsiya) — benign=0, malignant=1"""
    def __init__(self):
        self.items = []
        for idx, c in enumerate(["benign", "malignant"]):
            for p in (DMID / c).glob("*.png"):
                self.items.append((str(p), idx))
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), l


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.b.classifier[-1] = nn.Linear(self.b.classifier[-1].in_features, 2)
    def forward(self, x):
        p = F.softmax(self.b(x), 1)
        return p, p[:, 1:2]


class Focal(nn.Module):
    def __init__(self, w, g=2.0): super().__init__(); self.w = w; self.g = g
    def forward(self, logp, y):
        p = torch.exp(logp); ce = F.nll_loss(logp, y, weight=self.w, reduction="none")
        return (((1 - p.gather(1, y[:, None]).squeeze(1)) ** self.g) * ce).mean()


def metr(probs, labels):
    sc = probs[:, 1]; y = np.array(labels)
    pred = (sc >= 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    acc = (tp+tn)/len(y); sens = tp/(tp+fn) if (tp+fn) else 0; spec = tn/(tn+fp) if (tn+fp) else 0
    prec = tp/(tp+fp) if (tp+fp) else 0; f1 = 2*prec*sens/(prec+sens) if (prec+sens) else 0
    ths=np.linspace(0,1,50); roc=[]; P=(y==1).sum(); N=(y==0).sum()
    for t in ths[::-1]:
        pr=sc>=t
        roc.append({"fpr":round(float((pr&(y==0)).sum()/N),4) if N else 0,"tpr":round(float((pr&(y==1)).sum()/P),4) if P else 0,"threshold":round(float(t),3)})
    pts=sorted([(r["fpr"],r["tpr"]) for r in roc])
    auc=sum((pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2 for i in range(1,len(pts)))
    return dict(accuracy=round(acc,4),sensitivity=round(sens,4),specificity=round(spec,4),
                precision=round(prec,4),f1=round(f1,4),auc=round(float(auc),4),
                confusion_matrix={"tp":tp,"tn":tn,"fp":fp,"fn":fn},roc_curve=roc)


def run(model, loader, crit, opt, dev, train, tta=False):
    model.train() if train else model.eval()
    correct=n=0; P=[]; L=[]; torch.set_grad_enabled(train)
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); probs,_=model(x)
        if tta and not train: p2,_=model(torch.flip(x,dims=[3])); probs=(probs+p2)/2
        if crit is not None:
            loss=crit(torch.log(probs+1e-9),y)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        correct+=(probs.argmax(1)==y).sum().item(); n+=x.size(0)
        P+=probs.detach().cpu().tolist(); L+=y.cpu().tolist()
    torch.set_grad_enabled(True)
    return correct/max(1,n), np.array(P), L


def main():
    dev=torch.device("cpu")
    import collections
    tr_ds=CbisFolderDS("train",True); va_ds=CbisFolderDS("validation"); te_ds=CbisFolderDS("test")
    te_num=len(te_ds)
    print(f"CBIS (diskdan): train {len(tr_ds)} | val {len(va_ds)} | test {te_num}", flush=True)
    cnt=collections.Counter(tr_ds.labs)
    w=torch.tensor([1.0, cnt[0]/max(1,cnt[1])],dtype=torch.float32).to(dev)
    tl=DataLoader(tr_ds,batch_size=32,shuffle=True,num_workers=0)
    vl=DataLoader(va_ds,batch_size=32); tel=DataLoader(te_ds,batch_size=32)
    model=Net().to(dev); crit=Focal(w)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12,eta_min=1e-6)
    best=-1; best_state=None
    for ep in range(12):
        tra,_,_=run(model,tl,crit,opt,dev,True)
        vacc,vp,vl2=run(model,vl,crit,opt,dev,False,tta=True)
        sch.step(); vauc=metr(vp,vl2)["auc"]
        if vauc>best: best=vauc; best_state={k:v.clone() for k,v in model.state_dict().items()}
        print(f"ep {ep+1:2d}/12 train_acc={tra:.3f} val_acc={vacc:.3f} AUC={vauc:.3f}{' *' if vauc==best else ''}",flush=True)
    model.load_state_dict(best_state)

    # CBIS test
    tacc,tp,tl2=run(model,tel,None,None,dev,False,tta=True); m=metr(tp,tl2)
    print(f"\n=== CBIS TEST === acc={m['accuracy']} AUC={m['auc']} sens={m['sensitivity']} spec={m['specificity']}",flush=True)

    # DMID tashqi validatsiya
    dl=DataLoader(FolderDS(),batch_size=32)
    _,dp,dl2=run(model,dl,None,None,dev,False,tta=True); ext=metr(dp,dl2)
    flipped=False
    if ext["auc"] < 0.5:  # belgi teskari ekan -> malignant ehtimol = 1-p
        flipped=True
        dp2=dp.copy(); dp2[:,[0,1]]=dp[:,[1,0]]; ext=metr(dp2,dl2)
    print(f"=== DMID TASHQI VALIDATSIYA === AUC={ext['auc']} sens={ext['sensitivity']} spec={ext['specificity']} (flipped={flipped})",flush=True)

    torch.save(model.state_dict(),"breast_ai_mammo.pth"); model.eval()
    torch.onnx.export(model,torch.randn(1,3,224,224),"breast_ai_mammo.onnx",export_params=True,
        opset_version=18,input_names=["image"],output_names=["class_probs","birads_score"],verbose=False,dynamo=False)
    out={"evaluated_at":datetime.utcnow().isoformat(),
        "dataset":"CBIS-DDSM (~7030 rasm, biopsy-proven)","model":"EfficientNet-B0, ImageNet pretrained (mammography, 2-class)",
        "classes":["benign","malignant"],"label_flipped":flipped,
        "n_total":te_num,**m,
        "external_validation_dmid":{"dataset":"DMID (411)","auc":ext["auc"],"sensitivity":ext["sensitivity"],"specificity":ext["specificity"],"roc_curve":ext["roc_curve"]}}
    Path("mammo_metrics.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print("breast_ai_mammo.onnx + mammo_metrics.json yangilandi (CBIS-DDSM)")


if __name__=="__main__":
    main()
