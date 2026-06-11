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

Dataset: BUSI (Al-Dhabyani 2020) — `Dataset_BUSI_with_GT/{benign,malignant,normal}/`.

### 1. Klassifikator o'qitish (benign/malignant)
```bash
python train_classifier.py busi_data/Dataset_BUSI_with_GT
# → breast_ai_model.onnx (yangi, haqiqiy model) + metrics.json (TEST to'plami)
```
MobileNetV3-Small, ImageNet pretrained, stratified 70/15/15 split (seed=42).
TEST to'plami o'qitishda ko'rilmaydi — metrics.json data leakage'siz halol.

### 2. Segmentatsiya U-Net o'qitish (ixtiyoriy)
```bash
python train_segmentation.py busi_data/Dataset_BUSI_with_GT
# → breast_ai_seg.onnx — /api/segment avtomatik ishlatadi
```

### 3. Faqat baholash (mavjud modelni test qilish)
```bash
python evaluate.py busi_data/Dataset_BUSI_with_GT  # → metrics.json
```

### Deploy
```bash
git add breast_ai_model.onnx breast_ai_seg.onnx metrics.json
git commit -m "Real BUSI-trained model + metrics" && git push
# Render avtomatik deploy, frontend Statistika sahifasida metrikalarni ko'rsatadi
```

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
