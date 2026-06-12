"""
Breast AI — To'liq eksperiment orkestratori
1) Arxitektura taqqoslovi (mobilenet/resnet/efficientnet)
2) Eng yaxshi arxitekturani tanlash (cancer AUC bo'yicha)
3) 5-fold cross-validation (95% CI)
4) Yakuniy modelni o'qitish + ONNX + metrics.json (CV va comparison bilan)

Ishlatish:  python run_experiments.py
"""
import subprocess, sys, json
from pathlib import Path

PY = sys.executable
ENV = {"PYTHONIOENCODING": "utf-8"}
import os
env = {**os.environ, **ENV}

def sh(args):
    print(f"\n{'='*60}\n>>> {' '.join(args)}\n{'='*60}", flush=True)
    r = subprocess.run([PY, "train.py"] + args, env=env)
    if r.returncode != 0:
        print(f"XATO: {args} (code {r.returncode})"); sys.exit(1)

# 1) Taqqoslov — har arxitektura 10 epoch
for arch in ["mobilenet_v3", "resnet18", "efficientnet_b0"]:
    sh(["--arch", arch, "--classes", "3", "--epochs", "10", "--compare"])

# 2) Eng yaxshi arxitektura
comp = json.loads(Path("comparison.json").read_text(encoding="utf-8"))
best = max(comp["results"], key=lambda r: (r["auc"], r["f1"]))["arch"]
print(f"\n### ENG YAXSHI ARXITEKTURA: {best} ###", flush=True)

# 3) 5-fold CV
sh(["--arch", best, "--classes", "3", "--folds", "5", "--epochs", "12"])

# 4) Yakuniy model (ko'p epoch) + eksport
sh(["--arch", best, "--classes", "3", "--epochs", "28", "--export"])

print("\n### BARCHA EKSPERIMENTLAR TUGADI ###", flush=True)
