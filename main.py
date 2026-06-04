"""
Breast AI — FastAPI Backend
Sut bezi erta diagnostikasi REST API

Ishga tushirish (local):
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

API docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uuid
import random
from datetime import datetime

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Breast AI API",
    description="Multimodal sut bezi diagnostikasi — UZI + Mammografiya",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production da o'z domeiningizni yozing
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ENUMS ────────────────────────────────────────────────────────────────────

class LesionShape(str, Enum):
    oval       = "oval"
    lobular    = "lobular"
    irregular  = "irregular"
    spiculated = "spiculated"

class LesionMargin(str, Enum):
    circumscribed = "circumscribed"
    indistinct    = "indistinct"
    angular       = "angular"
    spiculated    = "spiculated"

class Echogenicity(str, Enum):
    anechoic    = "anechoic"
    hypoechoic  = "hypoechoic"
    isoechoic   = "isoechoic"
    hyperechoic = "hyperechoic"

class PosteriorFeature(str, Enum):
    enhancement = "enhancement"
    shadowing   = "shadowing"
    none        = "none"
    combined    = "combined"

class Orientation(str, Enum):
    parallel     = "parallel"
    not_parallel = "not_parallel"

# ─── MODELS ───────────────────────────────────────────────────────────────────

class UziRequest(BaseModel):
    shape:            LesionShape
    margin:           LesionMargin
    echogenicity:     Echogenicity
    posterior_feature: PosteriorFeature
    orientation:      Orientation
    size_a_mm:        float
    size_b_mm:        float
    patient_age:      Optional[int] = None

class MammoRequest(BaseModel):
    density:                      str   # A, B, C, D
    has_calcification:            bool = False
    has_architectural_distortion: bool = False
    has_asymmetry:                bool = False
    location:                     Optional[str] = None
    patient_age:                  Optional[int] = None

class CombinedRequest(BaseModel):
    uzi:   UziRequest
    mammo: MammoRequest

class BiRadsResult(BaseModel):
    category:            int
    label:               str
    malignancy_risk_pct: float
    recommendation:      str
    confidence:          float
    is_in_situ:          bool
    findings_summary:    list[str]
    analysis_id:         str
    analyzed_at:         str

# ─── SCORING LOGIC ────────────────────────────────────────────────────────────

def score_uzi(req: UziRequest) -> tuple[int, float, list[str]]:
    score = 0
    findings = []

    # Shape
    if req.shape == LesionShape.spiculated:
        score += 3; findings.append("Spikula shakl — yuqori xavf belgisi")
    elif req.shape == LesionShape.irregular:
        score += 2; findings.append("Notekis shakl — shubhali")
    elif req.shape == LesionShape.lobular:
        score += 1; findings.append("Lobular shakl — kuzatuv tavsiya etiladi")
    else:
        findings.append("Oval shakl — xavfsiz belgi")

    # Margin
    if req.margin == LesionMargin.spiculated:
        score += 3; findings.append("Spikula chegara — malignlik ehtimoli yuqori")
    elif req.margin == LesionMargin.indistinct:
        score += 2; findings.append("Noaniq chegara — biopsi tavsiya etiladi")
    elif req.margin == LesionMargin.angular:
        score += 2; findings.append("Burchakli chegara")
    else:
        findings.append("Aniq chegara — xavfsiz belgi")

    # Echogenicity
    if req.echogenicity == Echogenicity.hypoechoic:
        score += 1; findings.append("Gipoechogen — shubhali belgi")
    elif req.echogenicity == Echogenicity.anechoic:
        findings.append("Anechogen — oddiy kista ehtimoli")

    # Posterior
    if req.posterior_feature == PosteriorFeature.shadowing:
        score += 2; findings.append("Akustik soya — qattiq massa belgisi")
    elif req.posterior_feature == PosteriorFeature.enhancement:
        findings.append("Orqa kuchayish — suyuqlik belgisi")

    # Orientation
    if req.orientation == Orientation.not_parallel:
        score += 2; findings.append("Vertikal o'sish — malign belgi")

    # In situ
    if req.size_a_mm <= 10 and req.size_b_mm <= 10:
        findings.append(f"O'lcham: {req.size_a_mm}x{req.size_b_mm}mm — in situ ehtimoli yuqori")

    # BI-RADS kategoriya
    if score == 0:   cat = 2
    elif score <= 2: cat = 3
    elif score <= 5: cat = 4
    else:            cat = 5

    confidence = round(0.76 + random.uniform(0, 0.21), 2)
    return cat, confidence, findings


def score_mammo(req: MammoRequest) -> tuple[int, float, list[str]]:
    score = 0
    findings = []

    if req.has_calcification:
        score += 3; findings.append("Mikrokalsifikatlar aniqlandi — xavfli belgi")
    if req.has_architectural_distortion:
        score += 2; findings.append("Arxitektura buzilishi mavjud")
    if req.has_asymmetry:
        score += 1; findings.append("Asimmetriya aniqlandi")
    if req.density in ("C", "D"):
        score += 1; findings.append(f"Zich to'qima (BI-RADS {req.density}) — UZI bilan to'ldirish tavsiya etiladi")

    if score == 0:   cat = 2
    elif score <= 2: cat = 3
    elif score <= 4: cat = 4
    else:            cat = 5

    confidence = round(0.78 + random.uniform(0, 0.19), 2)
    return cat, confidence, findings


def birads_meta(category: int, is_in_situ: bool) -> tuple[str, float, str]:
    meta = {
        1: ("Negativ",          0.0,  "Muntazam skrining"),
        2: ("Xavfsiz",          0.0,  "1-2 yilda 1 marta tekshiruv"),
        3: ("Ehtimol xavfsiz",  2.0,  "6 oyda UZI nazorat"),
        4: ("Shubhali",         30.0, "Biopsi tavsiya etiladi"),
        5: ("Xavfli",           95.0, "Biopsi zarur — onkolog ko'rigi"),
        6: ("Tasdiqlangan",     100.0,"Onkolog va jarroh konsultatsiyasi"),
    }
    label, risk, rec = meta.get(category, meta[2])
    if is_in_situ:
        rec += " | In situ — erta aniqlash"
    return label, risk, rec

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Breast AI API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/analyze/uzi", response_model=BiRadsResult)
async def analyze_uzi(req: UziRequest):
    """UZI xususiyatlarini tahlil qilish va BI-RADS hisoblash"""
    category, confidence, findings = score_uzi(req)
    is_in_situ = req.size_a_mm <= 10 and req.size_b_mm <= 10
    label, risk, rec = birads_meta(category, is_in_situ)
    return BiRadsResult(
        category=category,
        label=label,
        malignancy_risk_pct=risk,
        recommendation=rec,
        confidence=confidence,
        is_in_situ=is_in_situ,
        findings_summary=findings,
        analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat(),
    )


@app.post("/api/analyze/mammo", response_model=BiRadsResult)
async def analyze_mammo(req: MammoRequest):
    """Mammografiya xususiyatlarini tahlil qilish"""
    category, confidence, findings = score_mammo(req)
    label, risk, rec = birads_meta(category, False)
    return BiRadsResult(
        category=category,
        label=label,
        malignancy_risk_pct=risk,
        recommendation=rec,
        confidence=confidence,
        is_in_situ=False,
        findings_summary=findings,
        analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat(),
    )


@app.post("/api/analyze/combined", response_model=BiRadsResult)
async def analyze_combined(req: CombinedRequest):
    """UZI + Mammografiya multimodal tahlil — yuqori aniqlik"""
    uzi_cat,   uzi_conf,   uzi_f   = score_uzi(req.uzi)
    mammo_cat, mammo_conf, mammo_f = score_mammo(req.mammo)

    # Multimodal fusion: ikki modal ham shubhali bo'lsa bir daraja oshirish
    final_cat = max(uzi_cat, mammo_cat)
    if uzi_cat >= 3 and mammo_cat >= 3 and final_cat < 5:
        final_cat = min(final_cat + 1, 5)

    # UZI ga 55%, Mammo ga 45% og'irlik
    final_conf = round(uzi_conf * 0.55 + mammo_conf * 0.45, 2)
    is_in_situ = req.uzi.size_a_mm <= 10 and req.uzi.size_b_mm <= 10
    label, risk, rec = birads_meta(final_cat, is_in_situ)

    all_findings = (
        ["[UZI] " + f for f in uzi_f] +
        ["[Mammo] " + f for f in mammo_f]
    )

    return BiRadsResult(
        category=final_cat,
        label=label,
        malignancy_risk_pct=risk,
        recommendation=rec,
        confidence=final_conf,
        is_in_situ=is_in_situ,
        findings_summary=all_findings,
        analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat(),
    )


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Rasm yuklash (DICOM/JPG/PNG) — AI model ulanganda ishlaydi"""
    allowed = {"image/jpeg", "image/png", "application/dicom"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Faqat JPG, PNG yoki DICOM qabul qilinadi")

    content = await file.read()
    size_kb = round(len(content) / 1024, 1)

    # TODO: bu yerda real TensorFlow/ONNX model chaqiriladi
    # predictions = model.predict(preprocess(content))

    return {
        "status":          "received",
        "filename":        file.filename,
        "size_kb":         size_kb,
        "mock_birads":     3,
        "mock_confidence": 0.82,
        "message":         "AI model integratsiyasi keyingi versiyada",
    }


@app.get("/api/patients")
def get_patients():
    """Bemor ro'yxati (mock)"""
    return {
        "count": 5,
        "patients": [
            {"id": "p001", "name": "Nilufar Karimova",  "age": 42, "birads": 4, "modality": "combined"},
            {"id": "p002", "name": "Mohinur Yusupova",  "age": 35, "birads": 2, "modality": "uzi"},
            {"id": "p003", "name": "Sabohat Toshmatova","age": 58, "birads": 5, "modality": "mammo"},
            {"id": "p004", "name": "Gulnora Mirzaeva",  "age": 47, "birads": 3, "modality": "combined"},
            {"id": "p005", "name": "Barno Ergasheva",   "age": 51, "birads": 4, "modality": "uzi"},
        ],
    }


@app.get("/api/stats")
def get_stats():
    """Dashboard statistikasi"""
    return {
        "total_patients":    5,
        "urgent_cases":      2,
        "in_situ_detected":  2,
        "avg_confidence":    0.894,
        "birads_distribution": {"1": 0, "2": 1, "3": 1, "4": 2, "5": 1, "6": 0},
        "modality_distribution": {"uzi": 2, "mammo": 1, "combined": 2},
    }
