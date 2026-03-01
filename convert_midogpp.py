import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import tifffile as tiff

Image.MAX_IMAGE_PIXELS = None  # disable PIL bomb warning


def convert_image(img_path: Path, out_path: Path):
    suffix = img_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        # Use tifffile for large TIFFs

        arr = tiff.imread(img_path)

        # Handle grayscale TIFFs
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # Handle channel-first TIFFs
        if arr.shape[0] in (3, 4) and arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))

        arr = arr.astype(np.uint8, copy=False)

    else:
        # PIL is fine for JPEG/PNG
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

    # Make writable + CHW
    arr = arr.copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    torch.save(tensor, out_path)


def main():
    src_dir = Path("/home/student4/PycharmProjects/DATA/MIDOGpp_IMAGES")     # change
    dst_dir = Path("/home/student4/PycharmProjects/DOG_MD/data/midogpp")     # change
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    img_paths = [
        p for p in src_dir.iterdir()
        if p.suffix.lower() in exts
    ]

    print(f"Found {len(img_paths)} images")

    for p in tqdm(img_paths):
        out_path = dst_dir / (p.stem + ".pt")
        if out_path.exists():
            continue
        try:
            convert_image(p, out_path)
        except ValueError as e:
            print(f"failed for image at path: {p}")


if __name__ == "__main__":
    main()