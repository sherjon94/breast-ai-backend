# Breast AI — FastAPI Backend

## Fayllar

```
breast_ai_backend/
├── main.py           # Asosiy FastAPI ilovasi
├── requirements.txt  # Python paketlar
├── Procfile          # Railway uchun ishga tushirish buyrug'i
├── railway.json      # Railway konfiguratsiya
└── README.md
```

## API Endpointlar

| Method | URL | Tavsif |
|--------|-----|--------|
| GET | / | Server holati |
| GET | /health | Health check |
| GET | /docs | Swagger UI |
| POST | /api/analyze/uzi | UZI tahlil |
| POST | /api/analyze/mammo | Mammografiya tahlil |
| POST | /api/analyze/combined | Multimodal tahlil |
| POST | /api/analyze/image | Rasm yuklash |
| GET | /api/patients | Bemorlar ro'yxati |
| GET | /api/stats | Statistika |

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
