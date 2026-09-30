import zipfile
import xml.etree.ElementTree as ET
import json
import re
import sqlite3
import os
from pathlib import Path

DOCX_DEFAULT_PATH = r"c:\Users\DELL\Downloads\Telegram Desktop\tablitsa pervichka.docx"
DB_PATH = Path(__file__).parent / "breast_ai.db"
JSON_OUTPUT = Path(__file__).parent / "clinical_cases.json"
FRONTEND_DATA_DIR = Path(r"C:\Users\DELL\Desktop\breast-ai-app\src\data")
FRONTEND_JSON_OUTPUT = FRONTEND_DATA_DIR / "clinical_cases.json"

def parse_docx(docx_path=DOCX_DEFAULT_PATH):
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Fayl topilmadi: {docx_path}")
    
    with zipfile.ZipFile(docx_path) as z:
        xml_content = z.read("word/document.xml")

    root = ET.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rows = root.findall(".//w:tr", ns)

    cases = []
    for i, r in enumerate(rows):
        cells = r.findall(".//w:tc", ns)
        row_text = ["".join(c.itertext()).strip() for c in cells]
        if not any(row_text):
            continue
        if i == 0 or (row_text[0] == "N" and "F.I.Sh" in row_text):
            continue
        
        num_str = row_text[0] if len(row_text) > 0 else ""
        name = row_text[1] if len(row_text) > 1 else ""
        if not name.strip():
            continue
        
        age_str = row_text[2] if len(row_text) > 2 else ""
        age = None
        if age_str.isdigit():
            age = int(age_str)
        else:
            m = re.search(r"(\d{4})\s*y", name)
            if m:
                age = 2025 - int(m.group(1))
                name = re.sub(r"\s*\d{4}\s*y\.?", "", name).strip()
            elif re.search(r"\d+", age_str):
                age = int(re.search(r"\d+", age_str).group())

        utt = row_text[3] if len(row_text) > 3 else ""
        mammo = row_text[4] if len(row_text) > 4 else ""
        mskt = row_text[5] if len(row_text) > 5 else ""
        biopsy = row_text[6] if len(row_text) > 6 else ""
        ihc = row_text[7] if len(row_text) > 7 else ""

        # Joylashuvi (tomon)
        combined_text = (utt + " " + mskt).upper()
        if "ОМЖ" in combined_text:
            laterality = "Ikkala ko'krak (ОМЖ)"
        elif "ПМЖ" in combined_text:
            laterality = "O'ng ko'krak (ПМЖ)"
        elif "ЛМЖ" in combined_text:
            laterality = "Chap ko'krak (ЛМЖ)"
        else:
            laterality = "Noma'lum"

        # BI-RADS toifa
        birads_sub = "4"
        sub_m = re.search(r"BI-?RADS\s*(4[ABCabc]|5|3)", utt + " " + mammo, re.IGNORECASE)
        if sub_m:
            birads_sub = sub_m.group(1).upper()

        # Molekulyar subtip
        subtype = "Aniqlanmagan"
        ihc_upper = ihc.upper()
        if "TRIPLE-NEGATIVE" in ihc_upper or "ТРИПЛ" in ihc_upper:
            subtype = "Triple-negative"
        elif "LUMINAL B (HER2-POSITIVE)" in ihc_upper or ("LUMINAL B" in ihc_upper and "HER2" in ihc_upper and ("POSITIVE" in ihc_upper or "ПОЛОЖИТЕЛЬНЫЙ" in ihc_upper)):
            subtype = "Luminal B (HER2+)"
        elif "LUMINAL B" in ihc_upper:
            subtype = "Luminal B"
        elif "LUMINAL A" in ihc_upper:
            subtype = "Luminal A"
        elif "HER2" in ihc_upper and "+++" in ihc_upper:
            subtype = "HER2-enriched"
        elif "DCIS" in ihc_upper:
            subtype = "In situ (DCIS)"

        case_num = int(num_str) if num_str.isdigit() else len(cases) + 1
        case_id = f"PERV-{case_num:03d}"

        cases.append({
            "id": case_id,
            "case_number": case_num,
            "full_name": name,
            "age": age,
            "laterality": laterality,
            "birads_category": 4,
            "birads_subcategory": birads_sub,
            "molecular_subtype": subtype,
            "utt_findings": utt,
            "mammography_findings": mammo,
            "mskt_petkt_findings": mskt,
            "biopsy_result": biopsy,
            "ihc_result": ihc,
        })
    
    return cases

def init_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clinical_cases (
            id TEXT PRIMARY KEY,
            case_number INTEGER,
            full_name TEXT,
            age INTEGER,
            laterality TEXT,
            birads_category INTEGER,
            birads_subcategory TEXT,
            molecular_subtype TEXT,
            utt_findings TEXT,
            mammography_findings TEXT,
            mskt_petkt_findings TEXT,
            biopsy_result TEXT,
            ihc_result TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def save_to_db(cases):
    from main import db, IS_PG
    conn = db()
    try:
        init_table(conn)
        for c in cases:
            vals = (
                c["id"], c["case_number"], c["full_name"], c["age"], c["laterality"],
                c["birads_category"], c["birads_subcategory"], c["molecular_subtype"],
                c["utt_findings"], c["mammography_findings"], c["mskt_petkt_findings"],
                c["biopsy_result"], c["ihc_result"]
            )
            cols = ("id, case_number, full_name, age, laterality, birads_category, "
                    "birads_subcategory, molecular_subtype, utt_findings, mammography_findings, "
                    "mskt_petkt_findings, biopsy_result, ihc_result")
            if IS_PG:
                conn.execute(
                    f"INSERT INTO clinical_cases({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (id) DO UPDATE SET full_name=EXCLUDED.full_name, age=EXCLUDED.age, "
                    "laterality=EXCLUDED.laterality, birads_subcategory=EXCLUDED.birads_subcategory, "
                    "molecular_subtype=EXCLUDED.molecular_subtype, utt_findings=EXCLUDED.utt_findings, "
                    "mammography_findings=EXCLUDED.mammography_findings, mskt_petkt_findings=EXCLUDED.mskt_petkt_findings, "
                    "biopsy_result=EXCLUDED.biopsy_result, ihc_result=EXCLUDED.ihc_result",
                    vals
                )
            else:
                conn.execute(f"INSERT OR REPLACE INTO clinical_cases({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)
        conn.commit()
        print(f"[OK] {len(cases)} ta bemor ma'lumotlar bazasiga muvaffaqiyatli saqlandi.")
    finally:
        conn.close()

def main():
    cases = parse_docx()
    print(f"[OK] DOCX fayldan {len(cases)} ta bemor o'qildi.")

    # JSON faylga saqlash (backend)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"[OK] Backend JSON saqlandi: {JSON_OUTPUT}")

    # JSON faylga saqlash (frontend)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FRONTEND_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"[OK] Frontend JSON saqlandi: {FRONTEND_JSON_OUTPUT}")

    # Bazaga yozish
    save_to_db(cases)

if __name__ == "__main__":
    main()
