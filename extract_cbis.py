"""CBIS parquet -> diskka 256px PNG (bir martalik, tez o'qitish uchun)."""
import io
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

OUT = Path("cbis_img")
for split in ["train", "validation", "test"]:
    t = pq.read_table(f"cbis_data/{split}.parquet")
    imgs = t.column("image"); labs = t.column("label").to_pylist()
    for c in ["benign", "malignant"]:
        (OUT / split / c).mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(t.num_rows):
        b = imgs[i].as_py()["bytes"]; lab = int(labs[i])
        cls = "malignant" if lab == 1 else "benign"
        dst = OUT / split / cls / f"{i:05d}.png"
        if dst.exists():
            n += 1; continue
        try:
            im = Image.open(io.BytesIO(b)).convert("L")
            w, h = im.size; s = 256 / max(w, h)
            im = im.resize((max(1, int(w*s)), max(1, int(h*s))))
            im.convert("RGB").save(dst)
            n += 1
        except Exception:
            pass
        if n % 1000 == 0:
            print(f"  {split}: {n}/{t.num_rows}", flush=True)
    print(f"{split}: {n} ta saqlandi", flush=True)
print("CBIS diskka chiqarildi (cbis_img/)")
