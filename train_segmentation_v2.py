"""
Yaxshilangan segmentatsiya — Attention U-Net, 192px, ko'proq epoch, BCE+Dice loss.
Maqsad: Dice koeffitsientini eski 0.62 dan oshirish.

Ishlatish:  python train_segmentation_v2.py
Natija:     breast_ai_seg.onnx  (main.py /api/segment avtomatik ishlatadi)
"""
import random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SIZE = 192
ROOT = Path("busi_data/Dataset_BUSI_with_GT")


def pairs():
    out = []
    for c in ["benign", "malignant"]:
        for img in (ROOT / c).glob("*.png"):
            if "mask" in img.stem.lower():
                continue
            m = img.with_name(img.stem + "_mask.png")
            if m.exists():
                out.append((str(img), str(m)))
    random.shuffle(out)
    return out


class SegDS(Dataset):
    def __init__(self, data, train=False): self.data = data; self.train = train
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ip, mp = self.data[i]
        img = Image.open(ip).convert("L").resize((SIZE, SIZE))
        mask = Image.open(mp).convert("L").resize((SIZE, SIZE))
        x = np.array(img, np.float32) / 255.0
        y = (np.array(mask, np.float32) > 127).astype(np.float32)
        if self.train:
            if random.random() < 0.5: x = x[:, ::-1].copy(); y = y[:, ::-1].copy()
            if random.random() < 0.3: x = x[::-1, :].copy(); y = y[::-1, :].copy()
            if random.random() < 0.5: x = np.clip(x * random.uniform(0.8, 1.2), 0, 1)
        x = np.stack([x, x, x], 0)
        return torch.from_numpy(x), torch.from_numpy(y[None])


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class AttnGate(nn.Module):
    """Attention gate — skip connection'dagi muhim sohalarni kuchaytiradi"""
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(g_ch, inter, 1), nn.BatchNorm2d(inter))
        self.wx = nn.Sequential(nn.Conv2d(x_ch, inter, 1), nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        a = self.relu(self.wg(g) + self.wx(x))
        return x * self.psi(a)


class AttUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.MaxPool2d(2)
        self.d1 = ConvBlock(3, 32); self.d2 = ConvBlock(32, 64)
        self.d3 = ConvBlock(64, 128); self.d4 = ConvBlock(128, 256)
        self.bott = ConvBlock(256, 512)
        self.u4 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.a4 = AttnGate(256, 256, 128); self.c4 = ConvBlock(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.a3 = AttnGate(128, 128, 64); self.c3 = ConvBlock(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.a2 = AttnGate(64, 64, 32); self.c2 = ConvBlock(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2); self.a1 = AttnGate(32, 32, 16); self.c1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        s1 = self.d1(x); s2 = self.d2(self.p(s1)); s3 = self.d3(self.p(s2)); s4 = self.d4(self.p(s3))
        b = self.bott(self.p(s4))
        x = self.u4(b); x = self.c4(torch.cat([self.a4(x, s4), x], 1))
        x = self.u3(x); x = self.c3(torch.cat([self.a3(x, s3), x], 1))
        x = self.u2(x); x = self.c2(torch.cat([self.a2(x, s2), x], 1))
        x = self.u1(x); x = self.c1(torch.cat([self.a1(x, s1), x], 1))
        return torch.sigmoid(self.out(x))


def dice_loss(p, t, e=1.0):
    p = p.reshape(p.size(0), -1); t = t.reshape(t.size(0), -1)
    return 1 - ((2*(p*t).sum(1)+e)/(p.sum(1)+t.sum(1)+e)).mean()

def dice_score(p, t, e=1.0):
    p = (p > 0.5).float().reshape(p.size(0), -1); t = t.reshape(t.size(0), -1)
    return ((2*(p*t).sum(1)+e)/(p.sum(1)+t.sum(1)+e)).mean().item()


def main():
    data = pairs(); nv = int(len(data)*0.15); val, train = data[:nv], data[nv:]
    print(f"Train {len(train)} | Val {len(val)} | {SIZE}px Attention U-Net", flush=True)
    dev = torch.device("cpu")
    tl = DataLoader(SegDS(train, True), batch_size=12, shuffle=True)
    vl = DataLoader(SegDS(val), batch_size=12)
    model = AttUNet().to(dev); bce = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
    best = 0; best_state = None
    for ep in range(30):
        model.train(); tot = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev); p = model(x)
            loss = bce(p, y) + dice_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        model.eval(); ds = []
        with torch.no_grad():
            for x, y in vl: ds.append(dice_score(model(x.to(dev)), y.to(dev)))
        vd = float(np.mean(ds))
        if vd > best: best = vd; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep+1:2d}/30 loss={tot/len(tl):.3f} val_Dice={vd:.3f}{' *' if vd==best else ''}", flush=True)
    model.load_state_dict(best_state)
    print(f"\nEng yaxshi Val Dice: {best:.3f} (eski: 0.62)")
    torch.save(model.state_dict(), "breast_ai_seg_v2.pth")
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), "breast_ai_seg.onnx", export_params=True,
        opset_version=18, input_names=["image"], output_names=["mask"], verbose=False, dynamo=False)
    print(f"breast_ai_seg.onnx yangilandi ({SIZE}px Attention U-Net, Dice {best:.3f})")


if __name__ == "__main__":
    main()
