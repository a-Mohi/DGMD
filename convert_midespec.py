import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ===== CONFIG =====
SRC_DIR = Path("/home/student4/PycharmProjects/DATA/MiDeSeC/test images")
DST_DIR = Path("/home/student4/PycharmProjects/DOG_MD/data/midesec")
DST_DIR.mkdir(parents=True, exist_ok=True)

to_tensor = transforms.ToTensor()

# Get list first for tqdm
bmp_files = list(SRC_DIR.glob("*.bmp"))

for bmp_path in bmp_files:
    img = Image.open(bmp_path).convert("RGB")
    tensor = to_tensor(img)

    out_path = DST_DIR / (bmp_path.stem + ".pt")
    torch.save(tensor, out_path)

print("Done.")
