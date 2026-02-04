import os
import re
import glob
import zipfile
import datetime
import pandas as pd
import streamlit as st
from PIL import Image
import requests

# Google Drive auto-backup
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="🩺 Radiologist Clinical Validation – Synthetic Breast Ultrasound",
    layout="wide"
)

# -------------------------
# Paths
# -------------------------
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
MSK_DIR = os.path.join(DATA_DIR, "masks")
META_XLSX = os.path.join(DATA_DIR, "metadata.xlsx")

OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "clinical_validation_scores.csv")
OUT_XLSX = os.path.join(OUT_DIR, "clinical_validation_scores.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Dataset from GitHub Release (DIRECT LINK)
# -------------------------
DATA_ZIP_URL = "https://github.com/halima-fouadi/clinical-validation-us/releases/download/v1.0-data/data.zip"
DATA_ZIP = "data.zip"

# -------------------------
# Criteria
# -------------------------
CRITERIA = [
    "Lesion Shape Accuracy (0–5)",
    "Margin Definition (0–5)",
    "Echogenicity (0–5)",
    "Tissue Context (0–5)",
    "Size and Proportions (0–5)",
    "BI-RADS Consistency (0–5)",
    "Segmentation Mask Accuracy (0–5)",
    "Overall Realism (0–5)",
]

# -------------------------
# Helpers (dataset)
# -------------------------
def ensure_data_present():
    """Download & extract data.zip from GitHub Release if data/images is missing/empty."""
    if os.path.exists(IMG_DIR) and len(glob.glob(os.path.join(IMG_DIR, "*"))) > 0:
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    st.info("Downloading dataset from GitHub Release...")

    try:
        r = requests.get(DATA_ZIP_URL, stream=True, timeout=180)
    except Exception as e:
        st.error(f"Download error: {e}")
        st.stop()

    if r.status_code != 200:
        st.error(
            f"Failed to download data.zip (HTTP {r.status_code}).\n"
            f"Check that data.zip exists in Release Assets.\nURL: {DATA_ZIP_URL}"
        )
        st.stop()

    total_bytes = 0
    with open(DATA_ZIP, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    if total_bytes < 1024 * 1024:
        st.error(
            "Downloaded file is too small (likely HTML error page). "
            "Verify Release Assets contains data.zip."
        )
        st.stop()

    st.info("Extracting dataset...")
    try:
        with zipfile.ZipFile(DATA_ZIP, "r") as z:
            z.extractall(".")  # expects zip contains data/...
    except Exception as e:
        st.error(f"Zip extraction failed: {e}. Make sure zip contains a top folder named 'data/'.")
        st.stop()

    st.success("Dataset ready ✅")

def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def get_case_id(name):
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else None

def find_mask_for_image(image_name):
    """Find mask like case_tumor_XXXX.* in data/masks."""
    cid = get_case_id(image_name)
    if not cid:
        return ""
    candidates = glob.glob(os.path.join(MSK_DIR, f"case_tumor_{cid}.*"))
    if candidates:
        return candidates[0]
    alt = os.path.join(MSK_DIR, image_name)
    return alt if os.path.exists(alt) else ""

def load_metadata_xlsx(path):
    if not os.path.exists(path):
        st.error(f"metadata.xlsx not found: {path}")
        st.stop()

    df = pd.read_excel(path)

    required = ["case_id", "Shape", "Margin", "Echogenicity", "BIRADS", "Classification"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in metadata.xlsx: {missing}")
        st.stop()

    df["case_id"] = df["case_id"].astype(str).str.extract(r"(\d+)")[0].fillna(df["case_id"].astype(str))
    df["case_id"] = df["case_id"].apply(lambda x: str(x).zfill(4))
    return df

def build_prompt(row):
    return (
        f"Shape={row['Shape']}, "
        f"Margin={row['Margin']}, "
        f"Echogenicity={row['Echogenicity']}, "
        f"BIRADS={row['BIRADS']}, "
        f"Classification={row['Classification']}"
    )

# -------------------------
# Helpers (saving locally)
# -------------------------
def append_to_csv(row, csv_path):
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

def clamp_idx(image_paths):
    st.session_state.idx = max(0, min(st.session_state.idx, len(image_paths) - 1))

def get_resume_index_after_last_case(reviewer_id, image_paths, results_csv):
    """Resume after the last evaluated case by this reviewer."""
    if not reviewer_id or not os.path.exists(results_csv):
        return 0

    df = pd.read_csv(results_csv)
    needed = {"reviewer", "image_name", "timestamp"}
    if not needed.issubset(set(df.columns)):
        return 0

    df_r = df[df["reviewer"].astype(str) == str(reviewer_id)].copy()
    if df_r.empty:
        return 0

    df_r = df_r.sort_values("timestamp", ascending=True)
    last_image = str(df_r.iloc[-1]["image_name"])

    names = [os.path.basename(p) for p in image_paths]
    if last_image in names:
        last_idx = names.index(last_image)
        return min(last_idx + 1, len(image_paths) - 1)

    return 0

# -------------------------
# Google Drive auto-backup helpers (FIXED + DEBUG)
# -------------------------
def _service_account_info_from_secrets() -> dict:
    """
    Streamlit secrets returns a dict-like object.
    We convert it to a plain dict and ensure private_key has proper newlines.
    """
    if "google" not in st.secrets:
        raise RuntimeError("Missing [google] section in Streamlit Secrets.")

    info = dict(st.secrets["google"])

    # Ensure newline formatting (some users paste with literal \n)
    pk = info.get("private_key", "")
    if "\\n" in pk and "\n" not in pk:
        info["private_key"] = pk.replace("\\n", "\n")

    # Optional sanity checks
    required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "token_uri"]
    missing = [k for k in required_keys if k not in info or not str(info.get(k, "")).strip()]
    if missing:
        raise RuntimeError(f"Secrets [google] missing keys: {missing}")

    return info

def get_drive_service():
    sa_info = _service_account_info_from_secrets()
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def upload_or_update_file(service, folder_id, local_path, drive_filename, mime_type):
    if not folder_id:
        raise RuntimeError("Missing [gdrive].folder_id in Streamlit Secrets.")
    if not os.path.exists(local_path):
        raise RuntimeError(f"Local file not found: {local_path}")

    query = f"name='{drive_filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

    if files:
        file_id = files[0]["id"]
        service.files().update(fileId=file_id, media_body=media).execute()
        return "updated"
    else:
        file_metadata = {"name": drive_filename, "parents": [folder_id]}
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        return "created"

def backup_results_to_drive(reviewer_id: str):
    """
    - Converts local CSV -> XLSX
    - Uploads BOTH to Google Drive folder
    - Uses per-reviewer filenames to avoid overwriting (recommended)
    """
    if not os.path.exists(OUT_CSV):
        return ("skipped", "skipped")

    # Convert to XLSX
    df = pd.read_csv(OUT_CSV)
    df.to_excel(OUT_XLSX, index=False)

    if "gdrive" not in st.secrets or "folder_id" not in st.secrets["gdrive"]:
        raise RuntimeError("Missing [gdrive].folder_id in Streamlit Secrets.")

    folder_id = str(st.secrets["gdrive"]["folder_id"]).strip()
    service = get_drive_service()

    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", (reviewer_id or "unknown").strip())
    drive_csv = f"clinical_validation_{safe_id}.csv"
    drive_xlsx = f"clinical_validation_{safe_id}.xlsx"

    status_csv = upload_or_update_file(service, folder_id, OUT_CSV, drive_csv, "text/csv")
    status_xlsx = upload_or_update_file(
        service, folder_id, OUT_XLSX, drive_xlsx,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return status_csv, status_xlsx

def drive_debug_test():
    """
    Creates and uploads a tiny text file to confirm that Drive upload works.
    """
    if "gdrive" not in st.secrets or "folder_id" not in st.secrets["gdrive"]:
        raise RuntimeError("Missing [gdrive].folder_id in Streamlit Secrets.")

    folder_id = str(st.secrets["gdrive"]["folder_id"]).strip()
    service = get_drive_service()

    test_path = os.path.join(OUT_DIR, "STREAMLIT_DRIVE_TEST.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(f"Drive test OK: {datetime.datetime.now().isoformat()}")

    status = upload_or_update_file(
        service,
        folder_id,
        test_path,
        drive_filename="STREAMLIT_DRIVE_TEST.txt",
        mime_type="text/plain"
    )
    return status

# -------------------------
# Ensure dataset exists
# -------------------------
ensure_data_present()

# -------------------------
# Load data
# -------------------------
image_paths = list_images(IMG_DIR)
if not image_paths:
    st.error("No images found in data/images/. Check that data.zip extracted correctly.")
    st.stop()

meta_df = load_metadata_xlsx(META_XLSX)
prompt_by_case = {row["case_id"]: build_prompt(row) for _, row in meta_df.iterrows()}

# -------------------------
# Sidebar (reviewer + Drive debug)
# -------------------------
st.sidebar.header("Reviewer")
reviewer = st.sidebar.text_input("Reviewer name / ID", value="")

st.sidebar.divider()
st.sidebar.subheader("🔧 Google Drive Debug")
if st.sidebar.button("Test Drive Upload"):
    try:
        status = drive_debug_test()
        st.sidebar.success(f"✅ Drive test success ({status}). Check Drive for STREAMLIT_DRIVE_TEST.txt")
    except Exception as e:
        st.sidebar.error("❌ Drive test failed. Details:")
        st.sidebar.exception(e)

if "idx" not in st.session_state:
    st.session_state.idx = 0
if "last_reviewer" not in st.session_state:
    st.session_state.last_reviewer = ""

if reviewer and reviewer != st.session_state.last_reviewer:
    st.session_state.idx = get_resume_index_after_last_case(reviewer, image_paths, OUT_CSV)
    st.session_state.last_reviewer = reviewer
    clamp_idx(image_paths)
    st.rerun()

if reviewer and os.path.exists(OUT_CSV):
    dfp = pd.read_csv(OUT_CSV)
    if "reviewer" in dfp.columns:
        st.sidebar.info(f"Saved evaluations: {(dfp['reviewer'].astype(str) == str(reviewer)).sum()}")

# -------------------------
# Navigation
# -------------------------
clamp_idx(image_paths)

st.title("🩺 Radiologist Clinical Validation – Synthetic Breast Ultrasound")

nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])
with nav1:
    if st.button("⬅️ Prev"):
        st.session_state.idx -= 1
        clamp_idx(image_paths)
        st.rerun()
with nav2:
    if st.button("Next ➡️"):
        st.session_state.idx += 1
        clamp_idx(image_paths)
        st.rerun()
with nav3:
    st.markdown(f"### Case {st.session_state.idx + 1} / {len(image_paths)}")
with nav4:
    pick = st.number_input("Go to", min_value=1, max_value=len(image_paths), value=st.session_state.idx + 1)
    if int(pick - 1) != st.session_state.idx:
        st.session_state.idx = int(pick - 1)
        clamp_idx(image_paths)
        st.rerun()

# -------------------------
# Current sample
# -------------------------
img_path = image_paths[st.session_state.idx]
image_name = os.path.basename(img_path)

cid = get_case_id(image_name)
cid_norm = str(cid).zfill(4) if cid else ""

mask_path = find_mask_for_image(image_name)
prompt = prompt_by_case.get(cid_norm, "Prompt not found in metadata.xlsx")

# -------------------------
# Display (NO overlay)
# -------------------------
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("Ultrasound Image")
    st.image(Image.open(img_path), use_container_width=True)
    st.caption(image_name)

with col2:
    st.subheader("Segmentation Mask")
    if mask_path and os.path.exists(mask_path):
        st.image(Image.open(mask_path), use_container_width=True)
        st.caption(os.path.basename(mask_path))
    else:
        st.warning("No mask found (expected case_tumor_XXXX.* in data/masks).")

with col3:
    st.subheader("Prompt")
    st.text_area("Prompt", prompt, height=220, disabled=True)

# -------------------------
# Survey
# -------------------------
st.divider()
st.subheader("📝 Survey / Barème (0–5 per criterion)")
st.caption("0 = Unacceptable, 5 = Excellent. Total = /40.")

scores = {}
cols = st.columns(2)
for i, c in enumerate(CRITERIA):
    with cols[i % 2]:
        scores[c] = st.slider(c, 0, 5, 3, key=f"{reviewer}_{image_name}_{c}")

total = int(sum(scores.values()))
st.markdown(f"### ✅ Total Score: **{total} / 40**")

comment = st.text_area("Comments (optional)", key=f"{reviewer}_{image_name}_comment", height=120)
decision = st.selectbox(
    "Decision (Mandatory)",
    ["", "Not acceptable", "Acceptable with limitations", "Clinically acceptable", "Clinically excellent"],
    key=f"{reviewer}_{image_name}_decision"
)

# -------------------------
# Save evaluation -> auto NEXT + auto Google Drive backup (WITH REAL ERROR DISPLAY)
# -------------------------
if st.button("💾 Save evaluation"):
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "reviewer": reviewer,
        "case_id": cid_norm,
        "image_name": image_name,
        "image_path": img_path,
        "mask_path": mask_path if (mask_path and os.path.exists(mask_path)) else "",
        "prompt": prompt,
        **scores,
        "total_score_0_40": total,
        "decision": decision,
        "comments": comment,
    }

    append_to_csv(row, OUT_CSV)
    st.success("Saved locally ✅")

    # Auto-backup to Google Drive
    try:
        s1, s2 = backup_results_to_drive(reviewer)
        st.success(f"Auto-save to Google Drive ✅ (CSV: {s1}, XLSX: {s2})")
    except Exception as e:
        st.error("Google Drive auto-save failed ❌ (details below)")
        st.exception(e)

    # automatically go to next case
    st.session_state.idx += 1
    clamp_idx(image_paths)
    st.rerun()

# -------------------------
# Preview saved results
# -------------------------
st.divider()
st.subheader("📊 Saved evaluations (preview)")
if os.path.exists(OUT_CSV):
    st.dataframe(pd.read_csv(OUT_CSV), use_container_width=True)
else:
    st.write("No evaluations saved yet.")

# -------------------------
# NB – Medical Scoring Guidelines (Footer)
# -------------------------
st.divider()
st.markdown(
    """
🩻 **NB – Medical criteria defined by clinical experts (Radiologists)**

**Scoring Guide (per criterion, 0–5):**  
- **0 – Unacceptable:** Completely wrong or impossible  
- **1 – Poor:** Very unrealistic, major errors  
- **2 – Fair:** Somewhat correct but flawed  
- **3 – Good:** Mostly accurate, minor issues  
- **4 – Very Good:** Highly realistic  
- **5 – Excellent:** Perfectly realistic, matches real ultrasound  

**Total Score Interpretation:**  
- Not acceptable  
- Acceptable with limitations  
- Clinically acceptable  
- Clinically excellent  

**Validation Criteria Interpretation:**  
- **Lesion Shape Accuracy:** Does the lesion shape match the intended type (round, oval, irregular) and BI-RADS description?  
- **Margin Definition:** Are the margins realistic (circumscribed, indistinct, spiculated) and consistent with the prompt?  
- **Echogenicity:** Does the lesion have the correct internal echo pattern (hypoechoic, hyperechoic, heterogeneous) matching the prompt?  
- **Tissue Context:** Is the surrounding tissue (parenchyma, fat, ducts) realistic? No unnatural artifacts?  
- **Size and Proportions:** Are lesion size, axis lengths, and area consistent with the prompt values and normal breast anatomy?  
- **BI-RADS Consistency:** Does the image correctly reflect the assigned BI-RADS category (benign vs. suspicious vs. malignant)?  
- **Segmentation Mask Accuracy:** Does the mask accurately outline the lesion without cutting off or including extra areas?  
- **Overall Realism:** Does the image look like a real ultrasound? No obvious synthetic artifacts?
"""
)
