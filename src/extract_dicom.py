"""
Extract PNG images from CMMD DICOM files and create processed_data.csv
by merging with CMMD_clinicaldata_revision.xlsx.

Label scheme (5 classes):
  Benign, Luminal A, Luminal B, HER2-enriched, triple negative

Rows with missing label are dropped.
"""

import os
import sys
import csv
import glob
import numpy as np

try:
    import pydicom
except ImportError:
    print("pydicom not installed. Run: pip install pydicom")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("openpyxl not installed. Run: pip install openpyxl")
    sys.exit(1)


# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICOM_ROOT = os.path.join(BASE_DIR, "CMMD")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "processed_images")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed_data.csv")
CLINICAL_XLSX = os.path.join(BASE_DIR, "CMMD_clinicaldata_revision.xlsx")


# ── helpers ──────────────────────────────────────────────────────────────
def safe_get(ds, tag, default=""):
    val = getattr(ds, tag, default)
    return str(val).strip() if val is not None else default


def dicom_to_png(ds, out_path):
    """Convert DICOM pixel data to 16-bit PNG."""
    arr = ds.pixel_array.astype(np.float64)

    if safe_get(ds, "PhotometricInterpretation") == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        arr = (arr - lo) / (hi - lo) * 65535.0
    else:
        arr = np.zeros_like(arr)

    Image.fromarray(arr.astype(np.uint16), mode="I;16").save(out_path)


def load_clinical_data(xlsx_path):
    """
    Load the Excel and build a lookup dict keyed by (ID1, LeftRight).

    Returns dict[(patient_id, laterality)] -> {
        age, abnormality, classification, subtype, label
    }
    label = 'Benign' if classification is Benign,
            else subtype value for Malignant cases.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    # Expected: ID1, LeftRight, Age, number, abnormality, classification, subtype
    col = {name: i for i, name in enumerate(header)}

    lookup = {}       # (patient_id, laterality) -> record
    pid_index = {}    # patient_id -> [laterality, ...] for fallback
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid = str(row[col["ID1"]]).strip()
        lr = str(row[col["LeftRight"]]).strip()       # L or R
        age = row[col["Age"]]
        abnormality = row[col["abnormality"]] or ""
        classification = str(row[col["classification"]] or "").strip()
        subtype = str(row[col["subtype"]] or "").strip()

        # Build unified 5-class label
        if classification == "Benign":
            label = "Benign"
        elif classification == "Malignant" and subtype and subtype != "None":
            label = subtype          # Luminal A / Luminal B / HER2-enriched / triple negative
        else:
            label = ""               # will be dropped later

        record = {
            "laterality": lr,
            "age": age,
            "abnormality": abnormality,
            "classification": classification,
            "subtype": subtype if subtype != "None" else "",
            "label": label,
        }
        lookup[(pid, lr)] = record
        pid_index.setdefault(pid, []).append(record)

    wb.close()
    return lookup, pid_index


# ── main ─────────────────────────────────────────────────────────────────
def main():
    # 1. Load clinical data
    print(f"Loading clinical data from {CLINICAL_XLSX} ...")
    clinical, pid_index = load_clinical_data(CLINICAL_XLSX)
    print(f"  {len(clinical)} clinical records loaded.")

    # 2. Discover DICOM files
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    dcm_files = sorted(glob.glob(os.path.join(DICOM_ROOT, "**", "*.dcm"), recursive=True))
    total = len(dcm_files)
    print(f"Found {total} DICOM files.")

    if total == 0:
        print("No DICOM files found. Exiting.")
        return

    # 3. Extract ALL images to PNG & build CSV rows for valid labels only
    rows = []
    errors = []
    images_extracted = 0
    skipped_no_clinical = 0
    skipped_no_label = 0

    for idx, dcm_path in enumerate(dcm_files, 1):
        rel = os.path.relpath(dcm_path, DICOM_ROOT)
        patient_id = rel.split(os.sep)[0]                       # D1-0001

        dcm_stem = os.path.splitext(os.path.basename(dcm_path))[0]
        image_filename = f"{patient_id}_{dcm_stem}.png"
        out_path = os.path.join(OUTPUT_IMG_DIR, image_filename)
        image_relpath = os.path.relpath(out_path, BASE_DIR)

        try:
            ds = pydicom.dcmread(dcm_path)

            # Always extract image to PNG
            dicom_to_png(ds, out_path)
            images_extracted += 1

            # Get laterality from DICOM header
            laterality = safe_get(ds, "ImageLaterality")
            if not laterality:
                laterality = safe_get(ds, "Laterality")

            # Look up clinical record: try exact (pid, laterality) first,
            # then fallback to patient-only if that patient has a single entry
            clin = clinical.get((patient_id, laterality))
            if clin is None:
                entries = pid_index.get(patient_id, [])
                if len(entries) == 1:
                    clin = entries[0]
                    laterality = clin["laterality"]
                else:
                    skipped_no_clinical += 1
                    continue

            # Only add to CSV if label is valid
            if not clin["label"]:
                skipped_no_label += 1
                continue

            rows.append({
                "patient_id": patient_id,
                "image_filename": image_filename,
                "image_path": image_relpath,
                "laterality": laterality,
                "age": clin["age"],
                "abnormality": clin["abnormality"],
                "label": clin["label"],
                "view_position": safe_get(ds, "ViewPosition"),
                "rows": safe_get(ds, "Rows"),
                "columns": safe_get(ds, "Columns"),
                "manufacturer": safe_get(ds, "Manufacturer"),
            })

        except Exception as e:
            errors.append((dcm_path, str(e)))
            print(f"  [ERROR] {dcm_path}: {e}")

        if idx % 200 == 0 or idx == total:
            print(f"  Processed {idx}/{total} ...")

    # 4. Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} records to {OUTPUT_CSV}")
    else:
        print("\nNo records to save.")

    # Summary
    print(f"\nTotal images extracted to PNG: {images_extracted}")
    label_counts = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print("Label distribution in CSV:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt}")

    print(f"\nNot in CSV (no clinical match): {skipped_no_clinical}")
    print(f"Not in CSV (missing label):     {skipped_no_label}")
    if errors:
        print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
