# Breast AI — FastAPI Backend v3.0

## Fayllar

```
breast_ai_backend/
├── main.py              # Asosiy FastAPI ilovasi
├── evaluate.py          # Model baholash → metrics.json yaratadi
├── breast_ai_model.onnx # AI model (MobileNetV3, 2 klass: benign/malignant)
├── metrics.json         # (evaluate.py natijasi — git'ga qo'shing)
├── breast_ai_seg.onnx   # (ixtiyoriy) U-Net segmentatsiya modeli
├── requirements.txt     # Python paketlar
├── Procfile / railway.json
└── README.md
```

## API Endpointlar

| Method | URL | Tavsif |
|--------|-----|--------|
| GET | / | Server holati |
| GET | /health | Health check (AI model yuklanganmi) |
| GET | /docs | Swagger UI |
| POST | /api/analyze/uzi | UZI tahlil (BI-RADS 4a/4b/4c bilan) |
| POST | /api/analyze/mammo | Mammografiya tahlil |
| POST | /api/analyze/combined | Multimodal tahlil |
| POST | /api/analyze/image | Rasm yuklash (JPG/PNG/DICOM/ZIP) — AI inference |
| POST | /api/explain | 🔥 Diqqat xaritasi (occlusion sensitivity heatmap) |
| POST | /api/segment | ✂️ O'simta segmentatsiyasi (Otsu / U-Net) |
| GET/POST/DELETE | /api/history | Tahlil tarixi (SQLite) |
| GET | /api/patients | Bemorlar (bazadan, real) |
| GET | /api/stats | Statistika (bazadan, real) |
| GET | /api/metrics | Model sifat ko'rsatkichlari (metrics.json) |

## Modelni o'qitish va baholash (PhD uchun muhim)

Dataset: BUSI (Al-Dhabyani 2020) — `Dataset_BUSI_with_GT/{normal,benign,malignant}/`.
Yuklab olish (Kaggle auth'siz): HuggingFace `gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`.

### To'liq eksperiment (tavsiya etiladi)
```bash
python run_experiments.py
# 1) Arxitektura taqqoslovi: MobileNetV3 / ResNet18 / EfficientNet-B0
# 2) Eng yaxshisini tanlaydi (cancer AUC bo'yicha)
# 3) 5-fold cross-validation (95% CI)
# 4) Yakuniy model + ONNX + metrics.json (CV va taqqoslov bilan)
```

### Yagona `train.py` (moslashuvchan)
```bash
# Yakuniy model (3-klass: normal/benign/malignant) + ONNX + metrics.json:
python train.py --arch efficientnet_b0 --classes 3 --epochs 28 --export
# 5-fold cross-validation (95% CI):
python train.py --arch efficientnet_b0 --classes 3 --folds 5 --epochs 12
# Arxitektura taqqoslovi uchun bitta arxitektura:
python train.py --arch resnet18 --classes 3 --epochs 10 --compare
```
Xususiyatlari: focal loss, augmentatsiya, test-time augmentation (TTA),
stratified split (seed=42, test held-out), ONNX eksport (eski interfeysga mos).
Model 2 yoki 3 klassli bo'lishi mumkin — backend chiqishdan avtomatik aniqlaydi.

### Segmentatsiya U-Net (ixtiyoriy)
```bash
python train_segmentation.py busi_data/Dataset_BUSI_with_GT
# → breast_ai_seg.onnx — /api/segment avtomatik ishlatadi
```

### Deploy
```bash
git add breast_ai_model.onnx breast_ai_seg.onnx metrics.json main.py
git commit -m "Real BUSI-trained model + metrics" && git push
# Render avtomatik deploy, frontend Statistika sahifasida metrikalarni ko'rsatadi
```

## Doimiy baza — PostgreSQL (v3.2)

Standart holatda SQLite ishlatiladi (lokal/efemer). `DATABASE_URL` muhit
o'zgaruvchisi berilsa, avtomatik **PostgreSQL** (doimiy) ga o'tadi — hisoblar
va tarix redeploy'da saqlanadi.

### Render'da Postgres ulash (bepul)
1. Render dashboard → **New** → **PostgreSQL** → bepul tarif → yarating
2. Yaratilgan bazaning **Internal Database URL** ini nusxalang
3. Backend web service → **Environment** → yangi o'zgaruvchi:
   `DATABASE_URL = <internal database url>`
4. Saqlang — avtomatik redeploy bo'ladi. Endi hisoblar doimiy saqlanadi.

Lokal test (Docker): `docker run -e POSTGRES_PASSWORD=pass -p 5432:5432 -d postgres`
keyin `DATABASE_URL=postgresql://postgres:pass@localhost:5432/postgres uvicorn main:app`.

## Yangi imkoniyatlar (v3.1)

- **3-klass model** (normal/benign/malignant) — backend adaptiv (2 yoki 3 klass)
- **Operating point** — `/api/analyze/image?threshold=0.3` (skrining: yuqori sezgirlik) yoki `0.7` (tasdiqlash: yuqori spesifiklik)
- **Cross-validation + 95% CI** va **arxitektura taqqoslovi** — `/api/metrics` qaytaradi, frontend ko'rsatadi
- **Ko'p shifokor** — `?doctor=<id>` filtri (`/api/history`, `/api/patients`, `/api/stats`); har shifokor o'z bemorlarini ko'radi

## Local ishga tushirish

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# http://localhost:8000/docs
```

## Railway.app ga yuklash

1. railway.app → "New Project" → "Deploy from GitHub repo"
2. Backend repository ni tanlang
3. Avtomatik deploy bo'ladi
4. Berilgan URL ni React ilovaga qo'shing

## React ilovaga ulash

`src/App.js` da API URL ni yangilang:

```javascript
const API_URL = "https://your-app.railway.app";

// Misol: UZI tahlil
const response = await fetch(`${API_URL}/api/analyze/uzi`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    shape: "irregular",
    margin: "indistinct",
    echogenicity: "hypoechoic",
    posterior_feature: "shadowing",
    orientation: "not_parallel",
    size_a_mm: 8.4,
    size_b_mm: 6.1
  })
});
const result = await response.json();
console.log(result.category); // 4
console.log(result.confidence); // 0.89
```
"# breast-ai-backend" 
