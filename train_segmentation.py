"""
Breast AI — U-Net segmentatsiya o'qitish (BUSI niqoblari bilan)
Kichik U-Net, 128x128, benign+malignant rasmlar va ularning maskalari.

Ishlatish:
    python train_segmentation.py busi_data/Dataset_BUSI_with_GT

Natija:
    breast_ai_seg.onnx  — main.py /api/segment avtomatik ishlatadi
                          (input: image[1,3,128,128], output: mask[1,1,128,128])
"""
import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SIZE = 128


def collect_with_masks(root: Path):
    pairs = []
    for c in ["benign", "malignant"]:
        for img in (root / c).glob("*.png"):
            if "mask" in img.stem.lower():
                continue
            mask = img.with_name(img.stem + "_mask.png")
            if mask.exists():
                pairs.append((str(img), str(mask)))
    random.shuffle(pairs)
    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, train=False):
        self.pairs = pairs; self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("L").resize((SIZE, SIZE))
        mask = Image.open(mp).convert("L").resize((SIZE, SIZE))
        x = np.array(img, np.float32) / 255.0
        y = (np.array(mask, np.float32) > 127).astype(np.float32)
        if self.train and random.random() < 0.5:
            x = x[:, ::-1].copy(); y = y[:, ::-1].copy()
        x = np.stack([x, x, x], 0)  # 3-kanal (RGB sifatida)
        return torch.from_numpy(x), torch.from_numpy(y[None])


class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 32); self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128); self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.c3 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.c2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2); self.c1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s1 = self.d1(x); s2 = self.d2(self.pool(s1)); s3 = self.d3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))
        x = self.c3(torch.cat([self.u3(b), s3], 1))
        x = self.c2(torch.cat([self.u2(x), s2], 1))
        x = self.c1(torch.cat([self.u1(x), s1], 1))
        return torch.sigmoid(self.out(x))


def dice_loss(pred, target, eps=1.0):
    pred = pred.reshape(pred.size(0), -1); target = target.reshape(target.size(0), -1)
    inter = (pred * target).sum(1)
    return 1 - ((2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)).mean()


def dice_score(pred, target, eps=1.0):
    pred = (pred > 0.5).float().reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    inter = (pred * target).sum(1)
    return ((2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)).mean().item()


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "busi_data/Dataset_BUSI_with_GT")
    pairs = collect_with_masks(root)
    n_val = int(len(pairs) * 0.15)
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"Train {len(train)} | Val {len(val)} (rasm+niqob juftliklari)")

    device = torch.device("cpu")
    tl = DataLoader(SegDataset(train, train=True), batch_size=16, shuffle=True)
    vl = DataLoader(SegDataset(val), batch_size=16)

    model = UNet().to(device)
    bce = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_dice, best_state = 0.0, None
    EPOCHS = 20
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            p = model(x)
            loss = bce(p, y) + dice_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        model.eval(); dices = []
        with torch.no_grad():
            for x, y in vl:
                dices.append(dice_score(model(x.to(device)), y.to(device)))
        vd = float(np.mean(dices))
        mark = ""
        if vd > best_dice:
            best_dice = vd; best_state = {k: v.clone() for k, v in model.state_dict().items()}; mark = " *BEST"
        print(f"Epoch {ep+1:2d}/{EPOCHS}  loss={tot/len(tl):.3f}  val_Dice={vd:.3f}{mark}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nEng yaxshi Val Dice: {best_dice:.3f}")

    torch.save(model.state_dict(), "breast_ai_seg.pth")
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, 3, SIZE, SIZE), "breast_ai_seg.onnx",
        export_params=True, opset_version=18,
        input_names=["image"], output_names=["mask"],
        verbose=False, dynamo=False,
    )
    print("breast_ai_seg.onnx yozildi — main.py /api/segment avtomatik ishlatadi")


if __name__ == "__main__":
    main()
