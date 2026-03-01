import shutil
from pathlib import Path

# ===== CONFIG =====
SRC_DIR = Path("/home/student4/PycharmProjects/DATA/MiDeSeC/train images")
DST_DIR = Path("/home/student4/PycharmProjects/DOG_MD/data/midesec")
DST_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(SRC_DIR.glob("*.csv"))

for csv_path in csv_files:
    dst_path = DST_DIR / csv_path.name
    shutil.copy(str(csv_path), str(dst_path))

print("Done.")
