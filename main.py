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
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import zipfile
import os

app = FastAPI(
    title="Breast AI API",
    description="Multimodal sut bezi diagnostikasi — UZI + Mammografiya + AI",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── AI MODEL ─────────────────────────────────────────────────────────────────

# DIQQAT: ONNX model 2 klassli — class_probs shape [1, 2] = [benign, malignant]
CLASSES = ["benign", "malignant"]

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# BI-RADS subkategoriyalar — malignant ehtimolga qarab
# 4a: past shubha (2-10%), 4b: o'rta (10-50%), 4c: yuqori (50-95%)
SUBCAT_META = {
    "2":  {"category": 2, "label": "Xavfsiz",                "risk": 0.0,  "rec": "1-2 yilda 1 marta tekshiruv"},
    "3":  {"category": 3, "label": "Ehtimol xavfsiz",        "risk": 2.0,  "rec": "6 oyda UZI nazorat"},
    "4a": {"category": 4, "label": "Past shubha (4a)",       "risk": 8.0,  "rec": "Biopsi tavsiya etiladi"},
    "4b": {"category": 4, "label": "O'rta shubha (4b)",      "risk": 30.0, "rec": "Yadro biopsiyasi tavsiya etiladi"},
    "4c": {"category": 4, "label": "Yuqori shubha (4c)",     "risk": 70.0, "rec": "Biopsi zarur — tezkor"},
    "5":  {"category": 5, "label": "Xavfli",                 "risk": 95.0, "rec": "Biopsi zarur — onkolog ko'rigi"},
}


def birads_from_probs(p_benign: float, p_malignant: float) -> str:
    """2 klassli model ehtimollaridan BI-RADS subkategoriya"""
    if p_malignant >= 0.97: return "5"
    if p_malignant >= 0.90: return "4c"
    if p_malignant >= 0.75: return "4b"
    if p_malignant >= 0.50: return "4a"
    if p_malignant >= 0.10: return "3"
    return "2"


# ONNX model yuklash
AI_AVAILABLE = False
ort_session = None

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
        print("[OK] AI model yuklandi:", str(MODEL_PATH))
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


# ─── SQLITE TARIX BAZASI ──────────────────────────────────────────────────────
# Eslatma: Render free tier diski efemer — redeploy da ma'lumot yo'qoladi.
# localStorage frontend'da asosiy zaxira bo'lib qoladi (ikki tomonlama sync).

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent / "breast_ai.db"))


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id TEXT PRIMARY KEY,
        created_at TEXT,
        patient_name TEXT,
        patient_age INTEGER,
        birads INTEGER,
        modality TEXT,
        confidence REAL,
        is_in_situ INTEGER,
        data TEXT)""")
    return conn


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
    """224x224 HWC array -> (p_benign, p_malignant, birads_score)"""
    outputs = ort_session.run(None, {"image": normalize_for_model(arr_hwc_255)})
    probs, birads_score = outputs
    return float(probs[0][0]), float(probs[0][1]), float(birads_score[0][0])


def build_ai_result(p_benign: float, p_malignant: float, birads_score: float) -> dict:
    pred_class = CLASSES[int(p_malignant >= p_benign)]
    confidence = max(p_benign, p_malignant)
    sub = birads_from_probs(p_benign, p_malignant)
    meta = SUBCAT_META[sub]

    return {
        "predicted_class":      pred_class,
        "confidence":           round(confidence, 4),
        "birads_category":      meta["category"],
        "birads_subcategory":   sub,
        "birads_label":         meta["label"],
        "malignancy_risk_pct":  meta["risk"],
        "recommendation":       meta["rec"],
        "is_in_situ":           False,  # o'lcham ma'lumotisiz aniqlab bo'lmaydi — frontend hisoblaydi
        "class_probabilities": {
            "benign":    round(p_benign, 4),
            "malignant": round(p_malignant, 4),
        },
        "birads_score":  round(birads_score, 4),
        "ai_model_used": True,
        "demo":          False,
        "analysis_id":   str(uuid.uuid4())[:8],
        "analyzed_at":   datetime.utcnow().isoformat(),
    }


def run_ai_inference(image_bytes: bytes) -> dict:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    p_b, p_m, score = infer_probs(np.array(img, dtype=np.float32))
    return build_ai_result(p_b, p_m, score)


def mock_inference() -> dict:
    """AI model yo'q bo'lganda DEMO natija — tasodifiy, klinik ahamiyatga ega EMAS"""
    p_m = round(random.uniform(0.05, 0.95), 4)
    p_b = round(1 - p_m, 4)
    sub = birads_from_probs(p_b, p_m)
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
        "birads_score":  round(random.uniform(0.2, 0.8), 4),
        "ai_model_used": False,
        "demo":          True,
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
    p_b, p_m, score = infer_probs(base224)
    base_result = build_ai_result(p_b, p_m, score)
    target_idx = 1 if p_m >= p_b else 0
    base_p = max(p_b, p_m)

    cell = 224 // grid
    heat = np.zeros((grid, grid), dtype=np.float32)
    mean_color = base224.mean(axis=(0, 1))

    for gy in range(grid):
        for gx in range(grid):
            occluded = base224.copy()
            y0, y1 = gy * cell, 224 if gy == grid - 1 else (gy + 1) * cell
            x0, x1 = gx * cell, 224 if gx == grid - 1 else (gx + 1) * cell
            occluded[y0:y1, x0:x1] = mean_color
            pb2, pm2, _ = infer_probs(occluded)
            p = pm2 if target_idx == 1 else pb2
            heat[gy, gx] = max(0.0, base_p - p)

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

    SIZE = 128
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
        "classes": CLASSES,
        "version": "3.0.0",
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
        analyzed_at=datetime.utcnow().isoformat())

@app.post("/api/analyze/mammo", response_model=BiRadsResult)
async def analyze_mammo(req: MammoRequest):
    cat, conf, findings, score = score_mammo(req)
    sub = subcat_from_score(cat, score)
    label, risk, rec = birads_meta(cat, False, sub)
    return BiRadsResult(category=cat, subcategory=sub, label=label, malignancy_risk_pct=risk,
        recommendation=rec, confidence=conf, is_in_situ=False,
        findings_summary=findings, analysis_id=str(uuid.uuid4())[:8],
        analyzed_at=datetime.utcnow().isoformat())

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
        analyzed_at=datetime.utcnow().isoformat())


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


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Rasm yuklash: JPG, PNG, DICOM (.dcm), ZIP arxiv"""
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
                    result = run_ai_inference(img_bytes) if AI_AVAILABLE else mock_inference()
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
        result = run_ai_inference(img_bytes) if AI_AVAILABLE else mock_inference()
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

    result = run_ai_inference(content) if AI_AVAILABLE else mock_inference()
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


# ─── TARIX (SQLITE) ───────────────────────────────────────────────────────────

@app.post("/api/history")
def save_history(record: dict):
    if not record.get("id"):
        record["id"] = str(uuid.uuid4())[:12]
    conn = db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO analyses(id, created_at, patient_name, patient_age, birads, modality, confidence, is_in_situ, data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(record["id"]),
                record.get("date") or datetime.utcnow().isoformat(),
                record.get("patientName") or "",
                int(record["patientAge"]) if str(record.get("patientAge") or "").isdigit() else None,
                int(record.get("birads") or 0),
                record.get("modality") or "",
                float(record.get("confidence") or 0),
                1 if record.get("isInSitu") else 0,
                json.dumps(record, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"saved": True, "id": record["id"]}


@app.get("/api/history")
def list_history(limit: int = 200):
    conn = db()
    try:
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
def clear_history():
    conn = db()
    try:
        cur = conn.execute("DELETE FROM analyses")
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()
    return {"deleted": deleted}


@app.get("/api/patients")
def get_patients():
    """Bazadagi real bemorlar — har bir bemor uchun oxirgi tahlil"""
    conn = db()
    try:
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
def get_stats():
    """Real statistika — SQLite bazadan hisoblanadi"""
    conn = db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        urgent = conn.execute("SELECT COUNT(*) FROM analyses WHERE birads>=4").fetchone()[0]
        in_situ = conn.execute("SELECT COUNT(*) FROM analyses WHERE is_in_situ=1").fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0] or 0
        birads_rows = conn.execute("SELECT birads, COUNT(*) FROM analyses GROUP BY birads").fetchall()
        modality_rows = conn.execute("SELECT modality, COUNT(*) FROM analyses GROUP BY modality").fetchall()
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
