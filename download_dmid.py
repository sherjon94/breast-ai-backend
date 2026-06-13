"""
DMID mammografiya rasmlarini yuklab, kichraytirib (384px PNG) saqlash.
Metadata'dan benign/malignant belgilarini oladi.
Natija: dmid_data/{benign,malignant}/*.png
"""
import urllib.request, io
from pathlib import Path
import openpyxl
from PIL import Image

REPO = "MyTwinLab/DMID_Breast_Cancer_Mammography_Dataset"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/TIFF%20Images/TIFF%20Images"
OUT = Path("dmid_data")
LABELMAP = {"M": "malignant", "B": "benign"}  # N (normal) tashlab yuboriladi

def main():
    wb = openpyxl.load_workbook("dmid_meta.xlsx"); ws = wb["Sheet1"]
    rows = [r for r in ws.iter_rows(values_only=True) if r and r[0] and "IMG" in str(r[0]).upper()]
    for c in LABELMAP.values():
        (OUT / c).mkdir(parents=True, exist_ok=True)
    ok = {"benign": 0, "malignant": 0}; fail = 0
    for r in rows:
        ref = str(r[0]).strip()             # "IMG001"
        lab = str(r[4]).strip() if r[4] else ""
        if lab not in LABELMAP:
            continue
        cls = LABELMAP[lab]
        dst = OUT / cls / f"{ref}.png"
        if dst.exists():
            ok[cls] += 1; continue
        url = f"{BASE}/{ref}.tif"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            img = Image.open(io.BytesIO(data)).convert("L")
            # kichraytirish (uzun tomoni 384px)
            w, h = img.size; s = 384 / max(w, h)
            img = img.resize((max(1, int(w*s)), max(1, int(h*s))))
            img.convert("RGB").save(dst)
            ok[cls] += 1
            if sum(ok.values()) % 50 == 0:
                print(f"  yuklandi: {ok}", flush=True)
        except Exception as e:
            fail += 1
    print(f"TUGADI — benign={ok['benign']} malignant={ok['malignant']} | xato={fail}")

if __name__ == "__main__":
    main()
