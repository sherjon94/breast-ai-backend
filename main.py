"""
Breast AI — FastAPI Backend v3.0
Real ONNX model + BI-RADS 4a/4b/4c + Explainability + Segmentatsiya + SQLite tarix
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uuid
import json
import random
import sqlite3
import base64
import hashlib
import secrets
import hmac
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import io
import zipfile
import os

app = FastAPI(
    title="Breast AI API",
    description="Multimodal sut bezi diagnostikasi — UZI + Mammografiya + AI",
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── AI MODEL ─────────────────────────────────────────────────────────────────

# Model 2 yoki 3 klassli bo'lishi mumkin — yuklashda avtomatik aniqlanadi.
# Klass tartibi: malignant DOIM oxirgi indeks.
#   2-klass: [benign, malignant]
#   3-klass: [normal, benign, malignant]
CLASSES = ["benign", "malignant"]

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# BI-RADS subkategoriyalar — malignant ehtimolga qarab
# 4a: past shubha (2-10%), 4b: o'rta (10-50%), 4c: yuqori (50-95%)
SUBCAT_META = {
    "1":  {"category": 1, "label": "Negativ (normal)",      "risk": 0.0,  "rec": "Muntazam skrining"},
    "2":  {"category": 2, "label": "Xavfsiz",                "risk": 0.0,  "rec": "1-2 yilda 1 marta tekshiruv"},
    "3":  {"category": 3, "label": "Ehtimol xavfsiz",        "risk": 2.0,  "rec": "6 oyda UZI nazorat"},
    "4a": {"category": 4, "label": "Past shubha (4a)",       "risk": 8.0,  "rec": "Biopsi tavsiya etiladi"},
    "4b": {"category": 4, "label": "O'rta shubha (4b)",      "risk": 30.0, "rec": "Yadro biopsiyasi tavsiya etiladi"},
    "4c": {"category": 4, "label": "Yuqori shubha (4c)",     "risk": 70.0, "rec": "Biopsi zarur — tezkor"},
    "5":  {"category": 5, "label": "Xavfli",                 "risk": 95.0, "rec": "Biopsi zarur — onkolog ko'rigi"},
}


def malignant_subcat(p_malignant: float) -> str:
    """Malignant ehtimolidan BI-RADS subkategoriya (4a/4b/4c)"""
    if p_malignant >= 0.97: return "5"
    if p_malignant >= 0.90: return "4c"
    if p_malignant >= 0.75: return "4b"
    if p_malignant >= 0.50: return "4a"
    if p_malignant >= 0.10: return "3"
    return "2"


# BI-RADS bo'yicha qayta ko'rik intervali (oy) — ACR ko'rsatmalariga yaqin
FOLLOWUP_MONTHS = {1: 12, 2: 12, 3: 6, "4a": 3, "4b": 1, "4c": 1, 4: 3, 5: 0, 6: 0}

def followup_recommendation(birads_category: int, subcat: str = None):
    """Qayta ko'rik tavsiyasi: necha oydan keyin va matn"""
    key = subcat if subcat in ("4a", "4b", "4c") else birads_category
    months = FOLLOWUP_MONTHS.get(key, FOLLOWUP_MONTHS.get(birads_category, 6))
    if months == 0:
        text = "Tezkor — biopsi/onkolog konsultatsiyasi zudlik bilan"
        next_date = datetime.utcnow().date().isoformat()
    else:
        text = f"{months} oydan keyin qayta ko'rik tavsiya etiladi"
        next_date = (datetime.utcnow() + timedelta(days=months * 30)).date().isoformat()
    return {"followup_months": months, "followup_text": text, "next_checkup_date": next_date}


# ONNX model yuklash
AI_AVAILABLE = False
ort_session = None
N_CLASSES = 2

try:
    import onnxruntime as ort
    env_path = os.environ.get("MODEL_PATH", "")
    possible_paths = [
        Path(env_path) if env_path else Path("nonexistent"),
        Path(__file__).parent / "breast_ai_model.onnx",
        Path("breast_ai_model.onnx"),
        Path("/opt/render/project/src/breast_ai_model.onnx"),
        Path("/opt/render/project/breast_ai_model.onnx"),
        Path("/app/breast_ai_model.onnx"),
    ]
    MODEL_PATH = None
    for p in possible_paths:
        if p.exists():
            MODEL_PATH = p
            break

    if MODEL_PATH:
        ort_session = ort.InferenceSession(str(MODEL_PATH))
        AI_AVAILABLE = True
        # Klass sonini model chiqishidan avtomatik aniqlash
        out_shape = ort_session.get_outputs()[0].shape
        N_CLASSES = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 2
        CLASSES = ["normal", "benign", "malignant"] if N_CLASSES == 3 else ["benign", "malignant"]
        print(f"[OK] AI model yuklandi ({N_CLASSES} klass: {CLASSES}):", str(MODEL_PATH))
    else:
        print("[!] ONNX model topilmadi — mock rejim")
except ImportError:
    print("[!] onnxruntime o'rnatilmagan — mock rejim")
except Exception as e:
    print(f"[!] Model yuklashda xato: {e} — mock rejim")

# Segmentatsiya modeli (ixtiyoriy — bo'lsa ishlatiladi)
SEG_AVAILABLE = False
seg_session = None
try:
    import onnxruntime as ort
    seg_path = Path(__file__).parent / "breast_ai_seg.onnx"
    if seg_path.exists():
        seg_session = ort.InferenceSession(str(seg_path))
        SEG_AVAILABLE = True
        print("[OK] Segmentatsiya modeli yuklandi")
except Exception as e:
    print(f"[!] Seg model: {e}")

# Mammografiya modeli (ixtiyoriy — DMID'da o'qitilgan, 2-klass benign/malignant)
MAMMO_AVAILABLE = False
mammo_session = None
try:
    import onnxruntime as ort
    mammo_path = Path(__file__).parent / "breast_ai_mammo.onnx"
    if mammo_path.exists():
        mammo_session = ort.InferenceSession(str(mammo_path))
        MAMMO_AVAILABLE = True
        print("[OK] Mammografiya modeli yuklandi")
except Exception as e:
    print(f"[!] Mammo model: {e}")


# ─── TARIX BAZASI (SQLite lokal / PostgreSQL Render) ──────────────────────────
# DATABASE_URL bo'lsa PostgreSQL (doimiy), bo'lmasa SQLite (lokal/efemer).

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent / "breast_ai.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
IS_PG = bool(DATABASE_URL)

if IS_PG:
    import psycopg

    class HybridRow:
        """sqlite3.Row kabi — ham pozitsion (r[0]), ham nomli (r['col']) kirish, dict(r) ishlaydi."""
        __slots__ = ("_c", "_v")
        def __init__(self, cols, vals): self._c = cols; self._v = list(vals)
        def keys(self): return self._c
        def __getitem__(self, k): return self._v[k] if isinstance(k, int) else self._v[self._c.index(k)]
        def __iter__(self): return iter(self._v)
        def __len__(self): return len(self._v)

    def _pg_rows(cursor):
        cols = [c.name for c in cursor.description] if cursor.description else []
        return lambda vals: HybridRow(cols, vals)

    class PGConn:
        """sqlite3.Connection interfeysiga mos wrapper (? -> %s tarjima)."""
        def __init__(self): self.conn = psycopg.connect(DATABASE_URL)
        def execute(self, sql, params=()):
            cur = self.conn.cursor(row_factory=_pg_rows)
            cur.execute(sql.replace("?", "%s"), params)
            return cur
        def commit(self): self.conn.commit()
        def close(self): self.conn.close()

_tables_ready = False

def _ensure_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id TEXT PRIMARY KEY, created_at TEXT, patient_name TEXT, patient_age INTEGER,
        birads INTEGER, modality TEXT, confidence REAL, is_in_situ INTEGER,
        doctor_id TEXT, data TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        id TEXT PRIMARY KEY, name TEXT, phone TEXT UNIQUE, specialization TEXT,
        clinic TEXT, license TEXT, password_hash TEXT, salt TEXT, role TEXT,
        approved INTEGER DEFAULT 0, token TEXT, created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS reports(
        id TEXT PRIMARY KEY, doctor_id TEXT, doctor_name TEXT, text TEXT,
        resolved INTEGER DEFAULT 0, created_at TEXT)""")
    if IS_PG:
        conn.execute("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS doctor_id TEXT")
    else:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()]
        if "doctor_id" not in cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN doctor_id TEXT")
    conn.commit()


def db():
    global _tables_ready
    if IS_PG:
        conn = PGConn()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    if not _tables_ready:
        _ensure_tables(conn)
        _tables_ready = True
    return conn


# ─── AUTH (rol asosida: admin / doctor) ──────────────────────────────────────

def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return h.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    h, _ = hash_password(password, salt)
    return hmac.compare_digest(h, password_hash)


def user_public(row: dict) -> dict:
    """Parolsiz foydalanuvchi ma'lumoti"""
    return {k: row[k] for k in ("id", "name", "phone", "specialization", "clinic", "license", "role", "approved", "created_at")}


def user_by_token(token: str):
    if not token:
        return None
    conn = db()
    try:
        r = conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def require_user(token: str):
    u = user_by_token(token)
    if not u:
        raise HTTPException(401, "Avtorizatsiya talab qilinadi — qaytadan kiring")
    if not u.get("approved"):
        raise HTTPException(403, "Hisobingiz hali admin tomonidan tasdiqlanmagan")
    return u


def require_admin(token: str):
    u = require_user(token)
    if u.get("role") != "admin":
        raise HTTPException(403, "Faqat admin uchun")
    return u


# ─── RASM O'QISH / VALIDATSIYA ────────────────────────────────────────────────

def looks_like_dicom(b: bytes) -> bool:
    """DICOM magic bytes: 128-baytdan keyin 'DICM'"""
    return len(b) > 132 and b[128:132] == b"DICM"


def read_dicom(dicom_bytes: bytes):
    """DICOM faylni o'qib PNG bytes + PIL Image qaytarish"""
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = ds.pixel_array.astype(np.float32)

        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
        pixel_array = pixel_array.astype(np.uint8)

        from PIL import Image
        if len(pixel_array.shape) == 2:
            img = Image.fromarray(pixel_array, mode='L').convert('RGB')
        elif len(pixel_array.shape) == 3:
            img = Image.fromarray(pixel_array).convert('RGB')
        else:
            raise ValueError("Noto'g'ri DICOM pixel format")

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue(), img
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"DICOM faylni o'qib bo'lmadi: {str(e)}")


def extract_zip_dicoms(zip_bytes: bytes):
    """ZIP arxivdan DICOM/rasm fayllarni chiqarish"""
    dicom_files = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith('.dcm') or lower.endswith('.dicom') or \
                   ('/' in name and not lower.endswith(('.jpg', '.png', '.txt', '.xml', '.json'))):
                    try:
                        data = zf.read(name)
                        if len(data) > 128:
                            dicom_files.append((name, data))
                    except Exception:
                        pass
                elif lower.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        data = zf.read(name)
                        dicom_files.append((name, data))
                    except Exception:
                        pass

        if not dicom_files:
            raise HTTPException(422, "ZIP arxivda DICOM yoki rasm fayllari topilmadi")

        return dicom_files
    except zipfile.BadZipFile:
        raise HTTPException(400, "Noto'g'ri ZIP fayl")


def is_medical_image(image_bytes: bytes):
    """Tibbiy rasm validatsiyasi — grayscale, qorong'i fon, yetarli kontrast"""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        if w < 50 or h < 50:
            return False, f"Rasm juda kichik ({w}x{h}px). Kamida 50x50 piksel kerak."

        img_small = img.resize((128, 128))
        arr = np.array(img_small, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        rg = np.mean(np.abs(r - g))
        rb = np.mean(np.abs(r - b))
        gb = np.mean(np.abs(g - b))
        colorfulness = (rg + rb + gb) / 3

        if colorfulness > 25:
            return False, (
                f"Bu rasm tibbiy emas (rang indeksi: {colorfulness:.1f}/25). "
                f"Iltimos faqat UZI yoki mammografiya rasmini yuklang."
            )

        brightness = np.mean(arr)
        if brightness > 200:
            return False, f"Rasm juda yorqin ({brightness:.0f}/200). UZI rasmlari odatda qorong'i fonli bo'ladi."

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        contrast = np.std(gray)
        if contrast < 10:
            return False, "Rasm kontrastsi juda past. Sifatli tibbiy rasm yuklang."

        white_ratio = np.mean(gray > 225)
        if white_ratio > 0.7:
            return False, "Rasm deyarli to'liq oq. UZI yoki mammografiya rasmi yuklang."

        return True, "OK"

    except Exception as e:
        return False, f"Rasmni o'qib bo'lmadi: {str(e)}"


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def normalize_for_model(arr_hwc_255: np.ndarray) -> np.ndarray:
    """HWC [0..255] float -> model kirishi NCHW"""
    x = arr_hwc_255 / 255.0
    x = (x - IMG_MEAN) / IMG_STD
    x = x.transpose(2, 0, 1)[None]
    return x.astype(np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    return normalize_for_model(np.array(img, dtype=np.float32))


def infer_probs(arr_hwc_255: np.ndarray):
    """224x224 HWC array -> (probs[N_CLASSES], birads_score). Malignant = oxirgi indeks."""
    outputs = ort_session.run(None, {"image": normalize_for_model(arr_hwc_255)})
    probs, birads_score = outputs
    return probs[0].astype(float), float(birads_score[0][0])


def build_ai_result(probs: np.ndarray, birads_score: float, threshold: float = 0.5) -> dict:
    p_malignant = float(probs[-1])
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]

    # 3-klassda aniq "normal" bashorat -> BI-RADS 1
    if len(CLASSES) == 3 and pred_class == "normal" and float(probs[0]) >= 0.5:
        sub = "1"
    else:
        sub = malignant_subcat(p_malignant)
    meta = SUBCAT_META[sub]

    return {
        "predicted_class":      pred_class,
        "confidence":           round(float(probs.max()), 4),
        "birads_category":      meta["category"],
        "birads_subcategory":   sub,
        "birads_label":         meta["label"],
        "malignancy_risk_pct":  meta["risk"],
        "recommendation":       meta["rec"],
        "is_in_situ":           False,  # o'lcham ma'lumotisiz aniqlab bo'lmaydi — frontend hisoblaydi
        "class_probabilities":  {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
        "operating_point":      threshold,
        "flagged_malignant":    bool(p_malignant >= threshold),
        **followup_recommendation(meta["category"], sub),
        "birads_score":  round(birads_score, 4),
        "ai_model_used": True,
        "demo":          False,
        "n_classes":     len(CLASSES),
        "analysis_id":   str(uuid.uuid4())[:8],
        "analyzed_at":   datetime.utcnow().isoformat(),
    }


def run_ai_inference(image_bytes: bytes, threshold: float = 0.5) -> dict:
    """TTA bilan: rasm + uning gorizontal aks-ko'rinishi o'rtachalanadi (barqarorroq ishonch)"""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    probs1, score1 = infer_probs(arr)
    probs2, score2 = infer_probs(arr[:, ::-1, :].copy())  # gorizontal flip (W o'qi)
    probs = (probs1 + probs2) / 2.0
    score = (score1 + score2) / 2.0
    result = build_ai_result(probs, score, threshold)
    result["tta"] = True
    return result


def run_mammo_inference(image_bytes: bytes, threshold: float = 0.5) -> dict:
    """Mammografiya modeli (DMID, 2-klass benign/malignant) + TTA"""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)

    def _one(a):
        x = a / 255.0
        x = (x - IMG_MEAN) / IMG_STD
        return mammo_session.run(None, {"image": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]

    probs = (_one(arr) + _one(arr[:, ::-1, :].copy())) / 2.0
    p_benign, p_malignant = float(probs[0]), float(probs[1])
    sub = malignant_subcat(p_malignant)
    meta = SUBCAT_META[sub]
    fu = followup_recommendation(meta["category"], sub)
    return {
        "predicted_class":     "malignant" if p_malignant >= p_benign else "benign",
        "confidence":          round(max(p_benign, p_malignant), 4),
        "birads_category":     meta["category"],
        "birads_subcategory":  sub,
        "birads_label":        meta["label"],
        "malignancy_risk_pct": meta["risk"],
        "recommendation":      meta["rec"],
        "is_in_situ":          False,
        "class_probabilities": {"benign": round(p_benign, 4), "malignant": round(p_malignant, 4)},
        "operating_point":     threshold,
        "flagged_malignant":   bool(p_malignant >= threshold),
        "ai_model_used":       True,
        "demo":                False,
        "modality_model":      "mammography",
        "n_classes":           2,
        "tta":                 True,
        "analysis_id":         str(uuid.uuid4())[:8],
        "analyzed_at":         datetime.utcnow().isoformat(),
        **fu,
    }


def mock_inference() -> dict:
    """AI model yo'q bo'lganda DEMO natija — tasodifiy, klinik ahamiyatga ega EMAS"""
    p_m = round(random.uniform(0.05, 0.95), 4)
    p_b = round(1 - p_m, 4)
    sub = malignant_subcat(p_m)
    meta = SUBCAT_META[sub]
    return {
        "predicted_class":     "malignant" if p_m >= 0.5 else "benign",
        "confidence":          max(p_b, p_m),
        "birads_category":     meta["category"],
        "birads_subcategory":  sub,
        "birads_label":        meta["label"] + " (DEMO)",
        "malignancy_risk_pct": meta["risk"],
        "recommendation":      "DEMO REJIM — bu natija tasodifiy, klinik qaror uchun ishlatmang!",
        "is_in_situ":          False,
        "class_probabilities": {"benign": p_b, "malignant": p_m},
        "operating_point":     0.5,
        "flagged_malignant":   bool(p_m >= 0.5),
        "birads_score":  round(random.uniform(0.2, 0.8), 4),
        "ai_model_used": False,
        "demo":          True,
        "n_classes":     len(CLASSES),
        "analysis_id":   str(uuid.uuid4())[:8],
        "analyzed_at":   datetime.utcnow().isoformat(),
    }


# ─── EXPLAINABILITY — OCCLUSION SENSITIVITY HEATMAP ──────────────────────────

def occlusion_heatmap(image_bytes: bytes, grid: int = 7) -> dict:
    """
    Occlusion sensitivity: rasmning har bir qismini berkitib, model
    ishonchining pasayishini o'lchaydi. Qaysi soha muhimligini ko'rsatadi.
    Gradient kerak emas — ONNX runtime bilan ishlaydi.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Katta rasmlarni cheklash (payload va tezlik uchun)
    if max(img.size) > 640:
        ratio = 640 / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    orig_w, orig_h = img.size

    base224 = np.array(img.resize((224, 224)), dtype=np.float32)
    probs, score = infer_probs(base224)
    base_result = build_ai_result(probs, score)
    target_idx = int(np.argmax(probs))   # bashorat qilingan klass
    base_p = float(probs[target_idx])

    cell = 224 // grid
    heat = np.zeros((grid, grid), dtype=np.float32)
    mean_color = base224.mean(axis=(0, 1))

    for gy in range(grid):
        for gx in range(grid):
            occluded = base224.copy()
            y0, y1 = gy * cell, 224 if gy == grid - 1 else (gy + 1) * cell
            x0, x1 = gx * cell, 224 if gx == grid - 1 else (gx + 1) * cell
            occluded[y0:y1, x0:x1] = mean_color
            probs2, _ = infer_probs(occluded)
            heat[gy, gx] = max(0.0, base_p - float(probs2[target_idx]))

    if heat.max() > 1e-9:
        heat = heat / heat.max()

    # Upsample + kolorizatsiya (qora -> qizil -> sariq)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode="L").resize((orig_w, orig_h), Image.BILINEAR)
    h = np.array(heat_img, dtype=np.float32) / 255.0
    base_rgb = np.array(img, dtype=np.float32)

    overlay = np.zeros_like(base_rgb)
    overlay[..., 0] = np.clip(h * 1.8, 0, 1) * 255          # qizil
    overlay[..., 1] = np.clip((h - 0.5) * 2, 0, 1) * 255    # sariq (yuqori muhimlikda)

    alpha = (0.55 * h)[..., None]
    blended = (base_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "heatmap_png_base64": heatmap_b64,
        "grid": grid,
        "method": "occlusion_sensitivity",
        "result": base_result,
    }


# ─── SEGMENTATSIYA ────────────────────────────────────────────────────────────

def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu usuli bilan optimal threshold"""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, thresh = 0.0, 0, 0.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var, thresh = var_between, t
    return float(thresh)


def _connected_components(mask: np.ndarray):
    """Oddiy flood-fill bilan bog'langan komponentlar (kichik rasm uchun)"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    components = []
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                current += 1
                stack = [(sy, sx)]
                labels[sy, sx] = current
                pixels = []
                while stack:
                    y, x = stack.pop()
                    pixels.append((y, x))
                    for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
                components.append((current, pixels))
    return labels, components


def unet_segmentation(image_bytes: bytes) -> dict:
    """U-Net ONNX modeli bilan segmentatsiya (breast_ai_seg.onnx)"""
    from PIL import Image

    _s = seg_session.get_inputs()[0].shape[-1]   # model kiritish o'lchamini avtomatik aniqlash
    SIZE = _s if isinstance(_s, int) else 128
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > 640:
        ratio = 640 / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    orig_w, orig_h = img.size

    gray = np.array(img.convert("L").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    x = np.stack([gray, gray, gray], 0)[None].astype(np.float32)
    out = seg_session.run(None, {"image": x})[0]  # [1,1,128,128]
    prob = out[0, 0]
    mask_small = prob > 0.5

    if mask_small.sum() < 10:
        return {"found": False, "method": "unet", "message": "O'simta aniqlanmadi"}

    # Original o'lchamga qaytarish
    mask_img = Image.fromarray((mask_small * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
    mask_big = np.array(mask_img) > 127

    # Kontur (qalin)
    interior = mask_big.copy()
    interior[1:, :] &= mask_big[:-1, :]; interior[:-1, :] &= mask_big[1:, :]
    interior[:, 1:] &= mask_big[:, :-1]; interior[:, :-1] &= mask_big[:, 1:]
    edge = mask_big & ~interior
    thick = edge.copy()
    thick[1:, :] |= edge[:-1, :]; thick[:-1, :] |= edge[1:, :]
    thick[:, 1:] |= edge[:, :-1]; thick[:, :-1] |= edge[:, 1:]

    overlay = np.array(img).copy()
    overlay[mask_big] = (overlay[mask_big] * 0.6 + np.array([46, 204, 113]) * 0.4).astype(np.uint8)
    overlay[thick] = [46, 204, 113]

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")

    ys, xs = np.where(mask_big)
    area_px = int(mask_big.sum())
    eq_diameter = float(2 * np.sqrt(area_px / np.pi))
    return {
        "found": True,
        "method": "unet",
        "approximate": False,
        "note": "U-Net model (BUSI niqoblarida o'qitilgan).",
        "overlay_png_base64": base64.b64encode(buf.getvalue()).decode(),
        "bbox_px": {"width": int(xs.max() - xs.min()), "height": int(ys.max() - ys.min())},
        "area_pct": round(area_px / (orig_w * orig_h) * 100, 2),
        "equivalent_diameter_px": round(eq_diameter, 1),
    }


def classical_segmentation(image_bytes: bytes) -> dict:
    """
    Klassik usul (taxminiy): UZI da o'simta odatda gipoechogen (qorong'i).
    Otsu threshold + eng katta markaziy qorong'i komponent.
    Aniq segmentatsiya uchun U-Net modelini breast_ai_seg.onnx sifatida qo'shing.
    """
    from PIL import Image, ImageFilter

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > 640:
        ratio = 640 / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    orig_w, orig_h = img.size

    SMALL = 160
    gray_img = img.convert("L").resize((SMALL, SMALL)).filter(ImageFilter.GaussianBlur(2))
    gray = np.array(gray_img, dtype=np.float32)

    t = _otsu_threshold(gray)
    mask = gray < t  # qorong'i hududlar

    labels, components = _connected_components(mask)
    if not components:
        return {"found": False, "message": "Qorong'i o'choq topilmadi"}

    # Chetga yopishgan katta fon hududlarini chiqarib tashlash
    candidates = []
    for cid, pixels in components:
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        touches_border = min(ys) == 0 or min(xs) == 0 or max(ys) == SMALL - 1 or max(xs) == SMALL - 1
        cy, cx = sum(ys) / len(ys), sum(xs) / len(xs)
        central = SMALL * 0.12 < cy < SMALL * 0.88 and SMALL * 0.12 < cx < SMALL * 0.88
        area = len(pixels)
        if area < 25 or area > SMALL * SMALL * 0.5:
            continue
        if touches_border and not central:
            continue
        candidates.append((area, cid, ys, xs))

    if not candidates:
        return {"found": False, "message": "Markaziy gipoechogen o'choq aniqlanmadi"}

    area, cid, ys, xs = max(candidates)
    comp_mask = (labels == cid)

    # Konturni topish (erosion farqi numpy shiftlar bilan)
    m = comp_mask
    interior = m.copy()
    interior[1:, :] &= m[:-1, :]
    interior[:-1, :] &= m[1:, :]
    interior[:, 1:] &= m[:, :-1]
    interior[:, :-1] &= m[:, 1:]
    edge = m & ~interior

    # Original o'lchamga qaytarish
    edge_img = Image.fromarray((edge * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
    edge_big = np.array(edge_img) > 127
    # Konturni qalinlashtirish
    thick = edge_big.copy()
    thick[1:, :] |= edge_big[:-1, :]
    thick[:-1, :] |= edge_big[1:, :]
    thick[:, 1:] |= edge_big[:, :-1]
    thick[:, :-1] |= edge_big[:, 1:]

    overlay = np.array(img).copy()
    overlay[thick] = [46, 204, 113]  # yashil kontur

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")

    # O'lchamlar (px da, kalibratsiyasiz — mm aniqlanmaydi)
    scale_x, scale_y = orig_w / SMALL, orig_h / SMALL
    bbox_w = (max(xs) - min(xs)) * scale_x
    bbox_h = (max(ys) - min(ys)) * scale_y
    area_pct = area / (SMALL * SMALL) * 100
    eq_diameter = float(2 * np.sqrt(area * scale_x * scale_y / np.pi))

    return {
        "found": True,
        "method": "classical_otsu",
        "approximate": True,
        "note": "Taxminiy klassik usul. Aniq segmentatsiya uchun U-Net model kerak (train_segmentation bo'limiga qarang).",
        "overlay_png_base64": base64.b64encode(buf.getvalue()).decode(),
        "bbox_px": {"width": round(bbox_w, 1), "height": round(bbox_h, 1)},
        "area_pct": round(area_pct, 2),
        "equivalent_diameter_px": round(eq_diameter, 1),
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
    subcategory: Optional[str] = None
    label: str
    malignancy_risk_pct: float
    recommendation: str
    confidence: float
    is_in_situ: bool
    findings_summary: list[str]
    analysis_id: str
    analyzed_at: str
    followup_months: Optional[int] = None
    followup_text: Optional[str] = None
    next_checkup_date: Optional[str] = None

class RegisterRequest(BaseModel):
    name: str
    phone: str
    password: str
    specialization: Optional[str] = ""
    clinic: Optional[str] = ""
    license: Optional[str] = ""

class LoginRequest(BaseModel):
    phone: str
    password: str

class ApproveRequest(BaseModel):
    token: str
    doctor_id: str
    approved: bool

class ReportRequest(BaseModel):
    token: str
    text: str

# ─── SCORING ──────────────────────────────────────────────────────────────────

def score_uzi(req: UziRequest):
    score, findings = 0, []
    if req.shape == LesionShape.spiculated:   score+=3; findings.append("Spikula shakl — yuqori xavf")
    elif req.shape == LesionShape.irregular:  score+=2; findings.append("Notekis shakl — shubhali")
    elif req.shape == LesionShape.lobular:    score+=1; findings.append("Lobular shakl — kuzatuv tavsiya")
    else:                                               findings.append("Oval shakl — xavfsiz")
    if req.margin == LesionMargin.spiculated: score+=3; findings.append("Spikula chegara — xavfli")
    elif req.margin == LesionMargin.indistinct: score+=2; findings.append("Noaniq chegara — biopsi tavsiya")
    elif req.margin == LesionMargin.angular:  score+=2; findings.append("Burchakli chegara — shubhali")
    if req.echogenicity == Echogenicity.hypoechoic: score+=1; findings.append("Gipoechogen — shubhali")
    if req.posterior_feature == PosteriorFeature.shadowing: score+=2; findings.append("Akustik soya")
    if req.orientation == Orientation.not_parallel: score+=2; findings.append("Vertikal o'sish — malign belgi")
    if req.size_a_mm<=10 and req.size_b_mm<=10: findings.append(f"{req.size_a_mm}x{req.size_b_mm}mm — in situ ehtimoli")
    cat = 2 if score==0 else 3 if score<=2 else 4 if score<=5 else 5
    conf={2:0.93,3:0.88,4:0.85,5:0.91}.get(cat,0.87)
    return cat, conf, findings, score

def score_mammo(req: MammoRequest):
    score, findings = 0, []
    if req.has_calcification: score+=3; findings.append("Mikrokalsifikatlar — xavfli")
    if req.has_architectural_distortion: score+=2; findings.append("Arxitektura buzilishi")
    if req.has_asymmetry: score+=1; findings.append("Asimmetriya")
    if req.density in("C","D"): score+=1; findings.append(f"Zich to'qima BI-RADS {req.density}")
    cat = 2 if score==0 else 3 if score<=2 else 4 if score<=4 else 5
    conf={2:0.94,3:0.89,4:0.86,5:0.92}.get(cat,0.88)
    return cat, conf, findings, score

def subcat_from_score(cat: int, score: int) -> Optional[str]:
    """Rule-based ball asosida 4a/4b/4c subkategoriya"""
    if cat != 4:
        return str(cat)
    if score <= 3: return "4a"
    if score <= 4: return "4b"
    return "4c"

def birads_meta(cat, in_situ, subcat=None):
    meta = {
        1:("Negativ",0,"Muntazam skrining"),
        2:("Xavfsiz",0,"1-2 yilda 1 marta"),
        3:("Ehtimol xavfsiz",2,"6 oyda UZI nazorat"),
        4:("Shubhali",30,"Biopsi tavsiya etiladi"),
        5:("Xavfli",95,"Biopsi zarur — onkolog ko'rigi"),
        6:("Tasdiqlangan",100,"Onkolog konsultatsiyasi"),
    }
    label, risk, rec = meta.get(cat, meta[2])
    if subcat in SUBCAT_META:
        m = SUBCAT_META[subcat]
        label, risk, rec = m["label"], m["risk"], m["rec"]
    if in_situ: rec += " | In situ aniqlandi"
    return label, risk, rec

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Breast AI API v3.0",
        "ai_model": "active" if AI_AVAILABLE else "mock",
        "endpoints": ["/api/analyze/uzi", "/api/analyze/mammo", "/api/analyze/combined",
                      "/api/analyze/image", "/api/explain", "/api/segment",
                      "/api/history", "/api/metrics", "/api/stats", "/api/patients"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_model_loaded": AI_AVAILABLE,
        "seg_model_loaded": SEG_AVAILABLE,
        "mammo_model_loaded": MAMMO_AVAILABLE,
        "classes": CLASSES,
        "n_classes": len(CLASSES),
        "version": "3.2.0",
        "db": "postgres" if IS_PG else "sqlite",
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post("/api/analyze/uzi", response_model=BiRadsResult)
async def analyze_uzi(req: UziRequest):
    cat, conf, findings, score = score_uzi(req)
    in_situ = req.size_a_mm<=10 and req.size_b_mm<=10
    sub = subcat_from_score(cat, score)
    label, risk, rec = birads_meta(cat, in_situ, sub)
    return BiRadsResult(category=cat, subcategory=sub, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=conf, is_in_situ=in_situ,
        findings_summary=findings, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat(), **followup_recommendation(cat, sub))

@app.post("/api/analyze/mammo", response_model=BiRadsResult)
async def analyze_mammo(req: MammoRequest):
    cat, conf, findings, score = score_mammo(req)
    sub = subcat_from_score(cat, score)
    label, risk, rec = birads_meta(cat, False, sub)
    return BiRadsResult(category=cat, subcategory=sub, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=conf, is_in_situ=False,
        findings_summary=findings, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat(), **followup_recommendation(cat, sub))

@app.post("/api/analyze/combined", response_model=BiRadsResult)
async def analyze_combined(req: CombinedRequest):
    uzi_cat, uzi_conf, uzi_f, uzi_s = score_uzi(req.uzi)
    mammo_cat, mammo_conf, mammo_f, mammo_s = score_mammo(req.mammo)
    final_cat = max(uzi_cat, mammo_cat)
    if uzi_cat>=3 and mammo_cat>=3 and final_cat<5:
        final_cat = min(final_cat+1, 5)
    final_conf = round(uzi_conf*0.55 + mammo_conf*0.45, 2)
    in_situ = req.uzi.size_a_mm<=10 and req.uzi.size_b_mm<=10
    sub = subcat_from_score(final_cat, max(uzi_s, mammo_s))
    label, risk, rec = birads_meta(final_cat, in_situ, sub)
    all_f = ["[UZI] "+f for f in uzi_f] + ["[Mammo] "+f for f in mammo_f]
    return BiRadsResult(category=final_cat, subcategory=sub, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=final_conf, is_in_situ=in_situ,
        findings_summary=all_f, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat(), **followup_recommendation(final_cat, sub))


async def _read_validated_image(file: UploadFile) -> bytes:
    """Upload faylni o'qib, DICOM bo'lsa konvert qilib, validatsiyadan o'tkazish"""
    content = await file.read()
    filename = (file.filename or "").lower()

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "Fayl hajmi 50MB dan oshmasin")

    if filename.endswith(('.dcm', '.dicom')) or looks_like_dicom(content):
        img_bytes, _ = read_dicom(content)
    else:
        img_bytes = content

    is_valid, reason = is_medical_image(img_bytes)
    if not is_valid:
        raise HTTPException(422, {"error": "Noto'g'ri rasm", "message": reason,
                                  "hint": "Iltimos, UZI yoki mammografiya rasmi yuklang."})
    return img_bytes


def _infer(img_bytes: bytes, threshold: float, modality: str = "uzi") -> dict:
    """Modallikka qarab to'g'ri modelni tanlash: mammo -> mammo modeli, aks holda US modeli"""
    if modality == "mammo" and MAMMO_AVAILABLE:
        return run_mammo_inference(img_bytes, threshold)
    return run_ai_inference(img_bytes, threshold) if AI_AVAILABLE else mock_inference()


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...), threshold: float = 0.5, modality: str = "uzi"):
    """Rasm yuklash: JPG, PNG, DICOM (.dcm), ZIP arxiv.
    threshold — operating point (skrining=0.3 yuqori sezgirlik, tasdiqlash=0.7 yuqori spesifiklik)
    modality — uzi/mammo/combined (mammo bo'lsa mammografiya modeli ishlatiladi)"""
    content = await file.read()
    filename = (file.filename or "").lower()
    content_type = file.content_type or ""

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "Fayl hajmi 50MB dan oshmasin")

    # ── ZIP arxiv ──
    if filename.endswith('.zip') or content_type == 'application/zip':
        dicom_list = extract_zip_dicoms(content)
        results = []

        for name, data in dicom_list[:5]:  # Max 5 ta fayl
            try:
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_bytes = data
                else:
                    img_bytes, _ = read_dicom(data)

                is_valid, reason = is_medical_image(img_bytes)
                if is_valid:
                    result = _infer(img_bytes, threshold, modality)
                    result["filename"] = name
                    results.append(result)
            except Exception:
                pass

        if not results:
            raise HTTPException(422, "ZIP ichidagi fayllarda tibbiy rasm topilmadi")

        best = max(results, key=lambda r: (r.get("birads_category", 0),
                                           r.get("class_probabilities", {}).get("malignant", 0)))
        best["zip_total_files"] = len(dicom_list)
        best["zip_analyzed"] = len(results)
        return best

    # ── DICOM fayl (kengaytma yoki magic bytes orqali) ──
    if filename.endswith(('.dcm', '.dicom')) or looks_like_dicom(content):
        img_bytes, _ = read_dicom(content)
        is_valid, reason = is_medical_image(img_bytes)
        if not is_valid:
            raise HTTPException(422, {"error": "DICOM rasm sifati past", "message": reason})
        result = _infer(img_bytes, threshold, modality)
        result["format"] = "DICOM"
        return result

    # ── Oddiy rasm (JPG/PNG) ──
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if content_type not in allowed and not filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(400, (
            "Qo'llab-quvvatlanadigan formatlar: JPG, PNG, DICOM (.dcm), ZIP arxiv. "
            f"Yuklangan fayl: {filename or content_type}"
        ))

    is_valid, reason = is_medical_image(content)
    if not is_valid:
        raise HTTPException(422, {"error": "Noto'g'ri rasm", "message": reason,
                                  "hint": "Iltimos, ultrasound (UZI) yoki mammografiya rasmi yuklang."})

    result = _infer(content, threshold, modality)
    result["format"] = "JPG/PNG"
    return result


@app.post("/api/explain")
async def explain_image(file: UploadFile = File(...)):
    """AI diqqat xaritasi (occlusion sensitivity heatmap)"""
    if not AI_AVAILABLE:
        raise HTTPException(503, "AI model yuklanmagan — heatmap faqat real model bilan ishlaydi")
    img_bytes = await _read_validated_image(file)
    return occlusion_heatmap(img_bytes)


@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    """O'simta segmentatsiyasi — U-Net model bo'lsa u, bo'lmasa klassik usul"""
    img_bytes = await _read_validated_image(file)
    # U-Net model bo'lsa undan, bo'lmasa klassik usuldan foydalanish
    if SEG_AVAILABLE:
        try:
            return unet_segmentation(img_bytes)
        except Exception as e:
            print(f"[!] U-Net segmentatsiya xatosi, klassik usulga o'tildi: {e}")
    return classical_segmentation(img_bytes)


# ─── AUTH ENDPOINTLAR ─────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    if len(req.password) < 4:
        raise HTTPException(400, "Parol kamida 4 belgidan iborat bo'lsin")
    conn = db()
    try:
        existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if conn.execute("SELECT 1 FROM users WHERE phone=?", (req.phone,)).fetchone():
            raise HTTPException(409, "Bu telefon raqami allaqachon ro'yxatdan o'tgan")
        # Birinchi foydalanuvchi = admin (avtomatik tasdiqlangan)
        is_admin = existing == 0
        ph, salt = hash_password(req.password)
        uid = str(uuid.uuid4())[:12]
        token = secrets.token_hex(24) if is_admin else None
        conn.execute(
            "INSERT INTO users(id,name,phone,specialization,clinic,license,password_hash,salt,role,approved,token,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (uid, req.name.strip(), req.phone.strip(), req.specialization, req.clinic, req.license,
             ph, salt, "admin" if is_admin else "doctor", 1 if is_admin else 0, token,
             datetime.utcnow().isoformat()))
        conn.commit()
    finally:
        conn.close()
    if is_admin:
        return {"registered": True, "role": "admin", "approved": True, "token": token,
                "user": {"id": uid, "name": req.name, "phone": req.phone, "role": "admin", "approved": 1},
                "message": "Admin sifatida ro'yxatdan o'tdingiz"}
    return {"registered": True, "role": "doctor", "approved": False,
            "message": "Ro'yxatdan o'tdingiz. Admin tasdiqlashini kuting."}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    conn = db()
    try:
        r = conn.execute("SELECT * FROM users WHERE phone=?", (req.phone.strip(),)).fetchone()
        if not r:
            raise HTTPException(401, "Telefon yoki parol noto'g'ri")
        u = dict(r)
        if not verify_password(req.password, u["password_hash"], u["salt"]):
            raise HTTPException(401, "Telefon yoki parol noto'g'ri")
        if not u["approved"]:
            raise HTTPException(403, "Hisobingiz hali admin tomonidan tasdiqlanmagan")
        token = secrets.token_hex(24)
        conn.execute("UPDATE users SET token=? WHERE id=?", (token, u["id"]))
        conn.commit()
    finally:
        conn.close()
    return {"token": token, "user": user_public(u)}


@app.get("/api/auth/me")
def auth_me(token: str):
    u = require_user(token)
    return {"user": user_public(u)}


@app.post("/api/auth/logout")
def logout(token: str):
    conn = db()
    try:
        conn.execute("UPDATE users SET token=NULL WHERE token=?", (token,))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


# ─── ADMIN ENDPOINTLAR ────────────────────────────────────────────────────────

@app.get("/api/admin/doctors")
def admin_doctors(token: str):
    require_admin(token)
    conn = db()
    try:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    return {"doctors": [user_public(dict(r)) for r in rows]}


@app.post("/api/admin/approve")
def admin_approve(req: ApproveRequest):
    require_admin(req.token)
    conn = db()
    try:
        conn.execute("UPDATE users SET approved=? WHERE id=?", (1 if req.approved else 0, req.doctor_id))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "doctor_id": req.doctor_id, "approved": req.approved}


# ─── MUAMMO BILDIRISH (REPORTS) ───────────────────────────────────────────────

@app.post("/api/report")
def submit_report(req: ReportRequest):
    u = require_user(req.token)
    if not req.text.strip():
        raise HTTPException(400, "Muammo matnini kiriting")
    conn = db()
    try:
        conn.execute("INSERT INTO reports(id, doctor_id, doctor_name, text, resolved, created_at) VALUES (?,?,?,?,?,?)",
                     (str(uuid.uuid4())[:12], u["id"], u.get("name") or "", req.text.strip(), 0, datetime.utcnow().isoformat()))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.get("/api/admin/reports")
def admin_reports(token: str):
    require_admin(token)
    conn = db()
    try:
        rows = conn.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 200").fetchall()
    finally:
        conn.close()
    return {"reports": [dict(r) for r in rows]}


@app.post("/api/admin/report-resolve")
def admin_report_resolve(req: ApproveRequest):
    # req.doctor_id = report_id, req.approved = resolved holati (ApproveRequest qayta ishlatildi)
    require_admin(req.token)
    conn = db()
    try:
        conn.execute("UPDATE reports SET resolved=? WHERE id=?", (1 if req.approved else 0, req.doctor_id))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


# ─── TARIX (SQLITE) ───────────────────────────────────────────────────────────

@app.post("/api/history")
def save_history(record: dict):
    if not record.get("id"):
        record["id"] = str(uuid.uuid4())[:12]
    conn = db()
    try:
        vals = (
            str(record["id"]),
            record.get("date") or datetime.utcnow().isoformat(),
            record.get("patientName") or "",
            int(record["patientAge"]) if str(record.get("patientAge") or "").isdigit() else None,
            int(record.get("birads") or 0),
            record.get("modality") or "",
            float(record.get("confidence") or 0),
            1 if record.get("isInSitu") else 0,
            record.get("doctorId") or "",
            json.dumps(record, ensure_ascii=False),
        )
        cols = "id, created_at, patient_name, patient_age, birads, modality, confidence, is_in_situ, doctor_id, data"
        if IS_PG:
            conn.execute(
                f"INSERT INTO analyses({cols}) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (id) DO UPDATE SET created_at=EXCLUDED.created_at, patient_name=EXCLUDED.patient_name, "
                "patient_age=EXCLUDED.patient_age, birads=EXCLUDED.birads, modality=EXCLUDED.modality, "
                "confidence=EXCLUDED.confidence, is_in_situ=EXCLUDED.is_in_situ, doctor_id=EXCLUDED.doctor_id, data=EXCLUDED.data",
                vals)
        else:
            conn.execute(f"INSERT OR REPLACE INTO analyses({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)", vals)
        conn.commit()
    finally:
        conn.close()
    return {"saved": True, "id": record["id"]}


def resolve_scope(token: Optional[str], doctor: Optional[str]) -> Optional[str]:
    """Rol asosida ko'rish doirasi: doctor -> faqat o'zi; admin -> ?doctor yoki barchasi."""
    if token:
        u = require_user(token)
        if u.get("role") != "admin":
            return u["id"]  # shifokor faqat o'z bemorlarini ko'radi
        return doctor  # admin: ?doctor bo'lsa filtr, bo'lmasa barchasi
    return doctor


@app.get("/api/history")
def list_history(limit: int = 200, doctor: Optional[str] = None, token: Optional[str] = None):
    doctor = resolve_scope(token, doctor)
    conn = db()
    try:
        if doctor:
            rows = conn.execute(
                "SELECT data FROM analyses WHERE doctor_id=? ORDER BY created_at DESC LIMIT ?",
                (doctor, min(limit, 500))).fetchall()
        else:
            rows = conn.execute(
                "SELECT data FROM analyses ORDER BY created_at DESC LIMIT ?", (min(limit, 500),)
            ).fetchall()
    finally:
        conn.close()
    records = []
    for (data,) in rows:
        try:
            records.append(json.loads(data))
        except Exception:
            pass
    return {"count": len(records), "records": records}


@app.delete("/api/history/{record_id}")
def delete_history_item(record_id: str):
    conn = db()
    try:
        cur = conn.execute("DELETE FROM analyses WHERE id=?", (record_id,))
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()
    return {"deleted": deleted}


@app.delete("/api/history")
def clear_history(doctor: Optional[str] = None, token: Optional[str] = None):
    doctor = resolve_scope(token, doctor)
    conn = db()
    try:
        if doctor:
            cur = conn.execute("DELETE FROM analyses WHERE doctor_id=?", (doctor,))
        else:
            cur = conn.execute("DELETE FROM analyses")
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()
    return {"deleted": deleted}


@app.get("/api/patients")
def get_patients(doctor: Optional[str] = None, token: Optional[str] = None):
    """Bazadagi real bemorlar — har bir bemor uchun oxirgi tahlil"""
    doctor = resolve_scope(token, doctor)
    conn = db()
    try:
        if doctor:
            rows = conn.execute(
                "SELECT patient_name, patient_age, birads, modality, MAX(created_at) "
                "FROM analyses WHERE patient_name != '' AND doctor_id=? GROUP BY patient_name "
                "ORDER BY MAX(created_at) DESC LIMIT 100", (doctor,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT patient_name, patient_age, birads, modality, MAX(created_at) "
                "FROM analyses WHERE patient_name != '' GROUP BY patient_name "
                "ORDER BY MAX(created_at) DESC LIMIT 100"
            ).fetchall()
    finally:
        conn.close()
    patients = [
        {"name": r[0], "age": r[1], "birads": r[2], "modality": r[3], "last_analysis": r[4]}
        for r in rows
    ]
    return {"count": len(patients), "patients": patients}


@app.get("/api/stats")
def get_stats(doctor: Optional[str] = None, token: Optional[str] = None):
    """Real statistika — SQLite bazadan hisoblanadi"""
    doctor = resolve_scope(token, doctor)
    conn = db()
    w = " WHERE doctor_id=?" if doctor else ""
    p = (doctor,) if doctor else ()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM analyses{w}", p).fetchone()[0]
        urgent = conn.execute(f"SELECT COUNT(*) FROM analyses WHERE birads>=4" + (" AND doctor_id=?" if doctor else ""), p).fetchone()[0]
        in_situ = conn.execute(f"SELECT COUNT(*) FROM analyses WHERE is_in_situ=1" + (" AND doctor_id=?" if doctor else ""), p).fetchone()[0]
        avg_conf = conn.execute(f"SELECT AVG(confidence) FROM analyses{w}", p).fetchone()[0] or 0
        birads_rows = conn.execute(f"SELECT birads, COUNT(*) FROM analyses{w} GROUP BY birads", p).fetchall()
        modality_rows = conn.execute(f"SELECT modality, COUNT(*) FROM analyses{w} GROUP BY modality", p).fetchall()
    finally:
        conn.close()
    return {
        "total_patients": total,
        "urgent_cases": urgent,
        "in_situ_detected": in_situ,
        "avg_confidence": round(avg_conf, 3),
        "ai_model_active": AI_AVAILABLE,
        "birads_distribution": {str(k): v for k, v in birads_rows},
        "modality_distribution": {k: v for k, v in modality_rows},
    }


# ─── MODEL METRIKALARI ────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics():
    """
    Model sifat ko'rsatkichlari — metrics.json dan o'qiladi.
    metrics.json ni yaratish: BUSI test to'plami bilan `python evaluate.py <dataset_dir>` ishga tushiring.
    """
    metrics_path = Path(__file__).parent / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            data["available"] = True
            return data
        except Exception as e:
            return {"available": False, "error": f"metrics.json o'qishda xato: {e}"}
    return {
        "available": False,
        "hint": "metrics.json topilmadi. BUSI test to'plami bilan 'python evaluate.py <dataset_papka>' ishga tushiring.",
    }


@app.get("/api/mammo-metrics")
def get_mammo_metrics():
    """Mammografiya modeli sifat ko'rsatkichlari (mammo_metrics.json)"""
    p = Path(__file__).parent / "mammo_metrics.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["available"] = True
            return d
        except Exception as e:
            return {"available": False, "error": str(e)}
    return {"available": False}
