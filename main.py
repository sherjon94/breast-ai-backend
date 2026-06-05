"""
Breast AI — FastAPI Backend v2.0
Real ONNX model + BI-RADS scoring
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uuid
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import io

app = FastAPI(
    title="Breast AI API",
    description="Multimodal sut bezi diagnostikasi — UZI + Mammografiya + AI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── AI MODEL ─────────────────────────────────────────────────────────────────

CLASSES = ["benign", "malignant", "normal"]
BIRADS_MAP = {
    "normal":    {"category": 1, "label": "Negativ",         "risk": 0.0,  "rec": "Muntazam skrining"},
    "benign":    {"category": 2, "label": "Xavfsiz",         "risk": 0.0,  "rec": "1-2 yilda 1 marta tekshiruv"},
    "malignant": {"category": 4, "label": "Shubhali",        "risk": 30.0, "rec": "Biopsi tavsiya etiladi"},
}

# ONNX model yuklash
AI_AVAILABLE = False
ort_session = None

try:
    import onnxruntime as ort
    MODEL_PATH = Path("breast_ai_model.onnx")
    if MODEL_PATH.exists():
        ort_session = ort.InferenceSession(str(MODEL_PATH))
        AI_AVAILABLE = True
        print("✓ AI model yuklandi:", str(MODEL_PATH))
    else:
        print("⚠ ONNX model topilmadi — mock rejim")
except ImportError:
    print("⚠ onnxruntime o'rnatilmagan — mock rejim")
except Exception as e:
    print(f"⚠ Model yuklashda xato: {e} — mock rejim")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Rasmni ONNX model uchun tayyorlash"""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)       # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr.astype(np.float32)


def run_ai_inference(image_bytes: bytes) -> dict:
    """ONNX model bilan inference"""
    img_array = preprocess_image(image_bytes)
    outputs = ort_session.run(None, {"image": img_array})
    probs, birads_score = outputs

    pred_class  = CLASSES[probs[0].argmax()]
    confidence  = float(probs[0].max())
    birads_info = BIRADS_MAP[pred_class]
    is_in_situ  = confidence > 0.85 and pred_class == "malignant"

    return {
        "predicted_class":      pred_class,
        "confidence":           round(confidence, 4),
        "birads_category":      birads_info["category"],
        "birads_label":         birads_info["label"],
        "malignancy_risk_pct":  birads_info["risk"],
        "recommendation":       birads_info["rec"],
        "is_in_situ":           is_in_situ,
        "class_probabilities": {
            "benign":    round(float(probs[0][0]), 4),
            "malignant": round(float(probs[0][1]), 4),
            "normal":    round(float(probs[0][2]), 4),
        },
        "birads_score":  round(float(birads_score[0][0]), 4),
        "ai_model_used": True,
        "analysis_id":   str(uuid.uuid4())[:8],
        "analyzed_at":   datetime.utcnow().isoformat(),
    }


def mock_inference() -> dict:
    """AI model yo'q bo'lganda mock natija"""
    cls   = random.choice(CLASSES)
    conf  = round(random.uniform(0.75, 0.96), 4)
    info  = BIRADS_MAP[cls]
    return {
        "predicted_class":     cls,
        "confidence":          conf,
        "birads_category":     info["category"],
        "birads_label":        info["label"],
        "malignancy_risk_pct": info["risk"],
        "recommendation":      info["rec"],
        "is_in_situ":          False,
        "class_probabilities": {
            "benign":    round(random.uniform(0.1, 0.8), 4),
            "malignant": round(random.uniform(0.1, 0.8), 4),
            "normal":    round(random.uniform(0.1, 0.8), 4),
        },
        "birads_score":  round(random.uniform(0.2, 0.8), 4),
        "ai_model_used": False,
        "analysis_id":   str(uuid.uuid4())[:8],
        "analyzed_at":   datetime.utcnow().isoformat(),
    }

# ─── ENUMS & MODELS ───────────────────────────────────────────────────────────

class LesionShape(str, Enum):
    oval="oval"; lobular="lobular"; irregular="irregular"; spiculated="spiculated"

class LesionMargin(str, Enum):
    circumscribed="circumscribed"; indistinct="indistinct"
    angular="angular"; spiculated="spiculated"

class Echogenicity(str, Enum):
    anechoic="anechoic"; hypoechoic="hypoechoic"
    isoechoic="isoechoic"; hyperechoic="hyperechoic"

class PosteriorFeature(str, Enum):
    enhancement="enhancement"; shadowing="shadowing"; none="none"; combined="combined"

class Orientation(str, Enum):
    parallel="parallel"; not_parallel="not_parallel"

class UziRequest(BaseModel):
    shape: LesionShape
    margin: LesionMargin
    echogenicity: Echogenicity
    posterior_feature: PosteriorFeature
    orientation: Orientation
    size_a_mm: float
    size_b_mm: float
    patient_age: Optional[int] = None

class MammoRequest(BaseModel):
    density: str
    has_calcification: bool = False
    has_architectural_distortion: bool = False
    has_asymmetry: bool = False
    location: Optional[str] = None
    patient_age: Optional[int] = None

class CombinedRequest(BaseModel):
    uzi: UziRequest
    mammo: MammoRequest

class BiRadsResult(BaseModel):
    category: int
    label: str
    malignancy_risk_pct: float
    recommendation: str
    confidence: float
    is_in_situ: bool
    findings_summary: list[str]
    analysis_id: str
    analyzed_at: str

# ─── SCORING ──────────────────────────────────────────────────────────────────

def score_uzi(req: UziRequest):
    score, findings = 0, []
    if req.shape == LesionShape.spiculated:   score+=3; findings.append("Spikula shakl — yuqori xavf")
    elif req.shape == LesionShape.irregular:  score+=2; findings.append("Notekis shakl — shubhali")
    elif req.shape == LesionShape.lobular:    score+=1; findings.append("Lobular shakl — kuzatuv tavsiya")
    else:                                               findings.append("Oval shakl — xavfsiz")
    if req.margin == LesionMargin.spiculated: score+=3; findings.append("Spikula chegara — xavfli")
    elif req.margin == LesionMargin.indistinct: score+=2; findings.append("Noaniq chegara — biopsi tavsiya")
    if req.echogenicity == Echogenicity.hypoechoic: score+=1; findings.append("Gipoechogen — shubhali")
    if req.posterior_feature == PosteriorFeature.shadowing: score+=2; findings.append("Akustik soya")
    if req.orientation == Orientation.not_parallel: score+=2; findings.append("Vertikal o'sish — malign belgi")
    if req.size_a_mm<=10 and req.size_b_mm<=10: findings.append(f"{req.size_a_mm}x{req.size_b_mm}mm — in situ ehtimoli")
    cat = 2 if score==0 else 3 if score<=2 else 4 if score<=5 else 5
    conf={2:0.93,3:0.88,4:0.85,5:0.91}.get(cat,0.87); return cat, conf, findings

def score_mammo(req: MammoRequest):
    score, findings = 0, []
    if req.has_calcification: score+=3; findings.append("Mikrokalsifikatlar — xavfli")
    if req.has_architectural_distortion: score+=2; findings.append("Arxitektura buzilishi")
    if req.has_asymmetry: score+=1; findings.append("Asimmetriya")
    if req.density in("C","D"): score+=1; findings.append(f"Zich to'qima BI-RADS {req.density}")
    cat = 2 if score==0 else 3 if score<=2 else 4 if score<=4 else 5
    conf={2:0.94,3:0.89,4:0.86,5:0.92}.get(cat,0.88); return cat, conf, findings

def birads_meta(cat, in_situ):
    meta = {
        1:("Negativ",0,"Muntazam skrining"),
        2:("Xavfsiz",0,"1-2 yilda 1 marta"),
        3:("Ehtimol xavfsiz",2,"6 oyda UZI nazorat"),
        4:("Shubhali",30,"Biopsi tavsiya etiladi"),
        5:("Xavfli",95,"Biopsi zarur — onkolog ko'rigi"),
        6:("Tasdiqlangan",100,"Onkolog konsultatsiyasi"),
    }
    label, risk, rec = meta.get(cat, meta[2])
    if in_situ: rec += " | In situ aniqlandi"
    return label, risk, rec

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Breast AI API v2.0",
        "ai_model": "active" if AI_AVAILABLE else "mock",
        "endpoints": ["/api/analyze/uzi", "/api/analyze/mammo",
                      "/api/analyze/combined", "/api/analyze/image"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_model_loaded": AI_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post("/api/analyze/uzi", response_model=BiRadsResult)
async def analyze_uzi(req: UziRequest):
    cat, conf, findings = score_uzi(req)
    in_situ = req.size_a_mm<=10 and req.size_b_mm<=10
    label, risk, rec = birads_meta(cat, in_situ)
    return BiRadsResult(category=cat, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=conf, is_in_situ=in_situ,
        findings_summary=findings, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat())

@app.post("/api/analyze/mammo", response_model=BiRadsResult)
async def analyze_mammo(req: MammoRequest):
    cat, conf, findings = score_mammo(req)
    label, risk, rec = birads_meta(cat, False)
    return BiRadsResult(category=cat, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=conf, is_in_situ=False,
        findings_summary=findings, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat())

@app.post("/api/analyze/combined", response_model=BiRadsResult)
async def analyze_combined(req: CombinedRequest):
    uzi_cat, uzi_conf, uzi_f = score_uzi(req.uzi)
    mammo_cat, mammo_conf, mammo_f = score_mammo(req.mammo)
    final_cat = max(uzi_cat, mammo_cat)
    if uzi_cat>=3 and mammo_cat>=3 and final_cat<5:
        final_cat = min(final_cat+1, 5)
    final_conf = round(uzi_conf*0.55 + mammo_conf*0.45, 2)
    in_situ = req.uzi.size_a_mm<=10 and req.uzi.size_b_mm<=10
    label, risk, rec = birads_meta(final_cat, in_situ)
    all_f = ["[UZI] "+f for f in uzi_f] + ["[Mammo] "+f for f in mammo_f]
    return BiRadsResult(category=final_cat, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=final_conf, is_in_situ=in_situ,
        findings_summary=all_f, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat())

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Rasm yuklash va AI tahlil (ONNX model bilan)"""
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Faqat JPG yoki PNG qabul qilinadi")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(400, "Fayl hajmi 20MB dan oshmasin")
    if AI_AVAILABLE:
        return run_ai_inference(content)
    else:
        return mock_inference()

@app.get("/api/patients")
def get_patients():
    return {"count": 5, "patients": [
        {"id":"p001","name":"Nilufar Karimova","age":42,"birads":4,"modality":"combined"},
        {"id":"p002","name":"Mohinur Yusupova","age":35,"birads":2,"modality":"uzi"},
        {"id":"p003","name":"Sabohat Toshmatova","age":58,"birads":5,"modality":"mammo"},
        {"id":"p004","name":"Gulnora Mirzaeva","age":47,"birads":3,"modality":"combined"},
        {"id":"p005","name":"Barno Ergasheva","age":51,"birads":4,"modality":"uzi"},
    ]}

@app.get("/api/stats")
def get_stats():
    return {
        "total_patients": 5,
        "urgent_cases": 2,
        "in_situ_detected": 2,
        "avg_confidence": 0.894,
        "ai_model_active": AI_AVAILABLE,
        "birads_distribution": {"1":0,"2":1,"3":1,"4":2,"5":1,"6":0},
        "modality_distribution": {"uzi":2,"mammo":1,"combined":2},
    }
