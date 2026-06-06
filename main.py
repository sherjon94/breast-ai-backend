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
import zipfile
import tempfile
import os

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


def read_dicom(dicom_bytes: bytes):
    """DICOM faylni o'qib PIL Image ga aylantirish"""
    try:
        import pydicom
        from pydicom.pixels import convert_color_space
        
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Normalize to 0-255
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        from PIL import Image
        # DICOM grayscale -> RGB
        if len(pixel_array.shape) == 2:
            img = Image.fromarray(pixel_array, mode='L').convert('RGB')
        elif len(pixel_array.shape) == 3:
            img = Image.fromarray(pixel_array).convert('RGB')
        else:
            raise ValueError("Noto'g'ri DICOM pixel format")
        
        # PNG ga convert qilib bytes qaytarish
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue(), img
    except Exception as e:
        raise HTTPException(422, f"DICOM faylni o'qib bo'lmadi: {str(e)}")


def extract_zip_dicoms(zip_bytes: bytes) :
    """ZIP arxivdan DICOM fayllarni chiqarish"""
    dicom_files = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                # DICOM fayllarni topish (.dcm yoki kengaytmasiz)
                if lower.endswith('.dcm') or lower.endswith('.dicom') or                    ('/' in name and not lower.endswith(('.jpg','.png','.txt','.xml','.json'))):
                    try:
                        data = zf.read(name)
                        if len(data) > 128:  # DICOM kamida 128 bayt
                            dicom_files.append((name, data))
                    except:
                        pass
                # Oddiy rasm fayllar
                elif lower.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        data = zf.read(name)
                        dicom_files.append((name, data))
                    except:
                        pass
        
        if not dicom_files:
            raise HTTPException(422, "ZIP arxivda DICOM yoki rasm fayllari topilmadi")
        
        return dicom_files
    except zipfile.BadZipFile:
        raise HTTPException(400, "Noto'g'ri ZIP fayl")


def is_medical_image(image_bytes: bytes):
    """
    Rasmning tibbiy ekanligini tekshirish.
    Ultrasound va mammografiya rasmlari odatda:
    - Ko'proq kulrang tonlarda bo'ladi
    - Past rang to'yinganligi (saturation)
    - Yuqori kontrast
    - Ko'pincha qora fon
    """
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_small = img.resize((64, 64))
        arr = np.array(img_small, dtype=np.float32)

        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # 1. Rang tekshirish — tibbiy rasmlar kulrang bo'ladi
        # RGB kanallari orasidagi farq kichik bo'lishi kerak
        rg_diff = np.mean(np.abs(r - g))
        rb_diff = np.mean(np.abs(r - b))
        gb_diff = np.mean(np.abs(g - b))
        avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

        # Agar rang farqi katta bo'lsa — rangli rasm (tibbiy emas)
        if avg_color_diff > 30:
            return False, f"Rasm tibbiy emas: rang to'yinganligi yuqori ({avg_color_diff:.1f}). Ultrasound yoki mammografiya rasmi yuklang."

        # 2. Qoralik tekshirish — tibbiy rasmlar ko'pincha qora fonga ega
        brightness = np.mean(arr)

        # Juda yorqin rasm (oddiy foto)
        if brightness > 220:
            return False, "Rasm juda yorqin. Ultrasound yoki mammografiya rasmi yuklang."

        # 3. Kontrast tekshirish — tibbiy rasmlar yuqori kontrastga ega
        gray = 0.299*r + 0.587*g + 0.114*b
        contrast = np.std(gray)

        if contrast < 15:
            return False, "Rasm kontrastsi past. Sifatli tibbiy rasm yuklang."

        # 4. O'lcham tekshirish
        w, h = img.size
        if w < 100 or h < 100:
            return False, f"Rasm o'lchami juda kichik ({w}x{h}). Kamida 100x100 piksel bo'lishi kerak."

        return True, "OK"

    except Exception as e:
        return False, f"Rasmni o'qib bo'lmadi: {str(e)}"


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
    """Rasm yuklash: JPG, PNG, DICOM (.dcm), ZIP arxiv"""
    content = await file.read()
    filename = (file.filename or "").lower()
    content_type = file.content_type or ""
    
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "Fayl hajmi 50MB dan oshmasin")

    # ── ZIP arxiv ──────────────────────────────────────────────────────────────
    if filename.endswith('.zip') or content_type == 'application/zip':
        dicom_list = extract_zip_dicoms(content)
        results = []
        
        for name, data in dicom_list[:5]:  # Max 5 ta fayl
            try:
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_bytes = data
                else:
                    img_bytes, _ = read_dicom(data)
                    img_bytes = img_bytes
                
                is_valid, reason = is_medical_image(img_bytes)
                if is_valid:
                    result = run_ai_inference(img_bytes) if AI_AVAILABLE else mock_inference()
                    result["filename"] = name
                    results.append(result)
            except:
                pass
        
        if not results:
            raise HTTPException(422, "ZIP ichidagi fayllarda tibbiy rasm topilmadi")
        
        # Eng xavfli natijani qaytarish
        best = max(results, key=lambda r: r.get("birads_category", 0))
        best["zip_total_files"] = len(dicom_list)
        best["zip_analyzed"] = len(results)
        return best

    # ── DICOM fayl ────────────────────────────────────────────────────────────
    if (filename.endswith('.dcm') or filename.endswith('.dicom') or
        content_type in ('application/dicom', 'application/octet-stream')):
        img_bytes, _ = read_dicom(content)
        is_valid, reason = is_medical_image(img_bytes)
        if not is_valid:
            raise HTTPException(422, {
                "error": "DICOM rasm sifati past",
                "message": reason,
            })
        result = run_ai_inference(img_bytes) if AI_AVAILABLE else mock_inference()
        result["format"] = "DICOM"
        return result

    # ── Oddiy rasm (JPG/PNG) ──────────────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if content_type not in allowed and not filename.endswith(('.jpg','.jpeg','.png')):
        raise HTTPException(400, (
            "Qo'llab-quvvatlanadigan formatlar: JPG, PNG, DICOM (.dcm), ZIP arxiv. "
            f"Yuklangan fayl: {filename or content_type}"
        ))

    is_valid, reason = is_medical_image(content)
    if not is_valid:
        raise HTTPException(422, {
            "error": "Noto'g'ri rasm",
            "message": reason,
            "hint": "Iltimos, ultrasound (UZI) yoki mammografiya rasmi yuklang."
        })

    result = run_ai_inference(content) if AI_AVAILABLE else mock_inference()
    result["format"] = "JPG/PNG"
    return result

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