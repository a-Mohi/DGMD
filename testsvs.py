import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_bbox_patch(
    svs_path,
    bbox,              # (x1, y1, x2, y2) in level 0 coords
    padding=50,        # pixels to extend on each side
    output_size=None   # set to 512 if you want resizing
):
    slide = openslide.OpenSlide(svs_path)

    slide_w, slide_h = slide.dimensions

    x1, y1, x2, y2 = bbox

    # ---- Extend bbox safely ----
    x1_ext = max(0, x1 - padding)
    y1_ext = max(0, y1 - padding)
    x2_ext = min(slide_w, x2 + padding)
    y2_ext = min(slide_h, y2 + padding)

    w_ext = x2_ext - x1_ext
    h_ext = y2_ext - y1_ext

    # ---- Read extended region from level 0 ----
    patch = slide.read_region(
        (x1_ext, y1_ext),
        0,
        (w_ext, h_ext)
    ).convert("RGB")

    # ---- Resize if needed ----
    if output_size is not None:
        scale_x = output_size / w_ext
        scale_y = output_size / h_ext
        patch = patch.resize((output_size, output_size), Image.BILINEAR)
    else:
        scale_x = scale_y = 1.0

    # ---- Compute original bbox location relative to extended patch ----
    rect_x = (x1 - x1_ext) * scale_x
    rect_y = (y1 - y1_ext) * scale_y
    rect_w = (x2 - x1) * scale_x
    rect_h = (y2 - y1) * scale_y

    # ---- Plot ----
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(patch)

    rect = patches.Rectangle(
        (rect_x, rect_y),
        rect_w,
        rect_h,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)
    ax.set_title(f"Original bbox inside padded patch")
    ax.axis("off")
    plt.show()

    slide.close()


svs_path = "/home/student4/PycharmProjects/DATA/MITOS_WSI_CCMCT_WSI/c3eb4b8382b470dd63a9.svs"

bbox = (7540,
        27932,
        7590,
        27982)  # example

visualize_bbox_patch(svs_path, bbox, output_size=512)