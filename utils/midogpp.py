from torch.utils.data import Dataset
import json
from pathlib import Path
import torch

class MIDOGpp(Dataset):
    """
    MIDOG++ Dataset loading dataset

    Classes:
        1 -> mitotic figure
        2 -> not mitotic figure (hard negative)
        background is implicit
    """

    def __init__(self, root_dir, json_path, transforms=None):
        """
        Args:
            root_dir (str | Path): directory containing images
            json_path (str | Path): path to annotation JSON
            transforms (callable, optional): image transforms only
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.source = "midogpp"
        with open(json_path, "r") as f:
            coco = json.load(f)

        # image_id -> image info
        self.images = {img["id"]: img for img in coco["images"]}

        # image_id -> list of annotations
        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        self.image_ids = sorted(self.images.keys())


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if (idx > 149 and idx < 201) or (idx == 69):
            idx += 51
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        img_path = self.root_dir / f'{img_info["file_name"][:-5]}.pt'
        image = torch.load(img_path, map_location="cpu", mmap=True)

        anns = self.annotations.get(image_id, [])

        points = []
        labels = []
        bboxes = []

        for ann in anns:
            category_id = ann["category_id"]
            if category_id not in (1, 2):
                continue

            # MIDOG bbox format is [x1, y1, x2, y2]
            x1, y1, x2, y2 = ann["bbox"]

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            points.append([cx, cy])
            labels.append(category_id)
            bboxes.append([x1, y1, x2, y2])

        # image = TF.to_tensor(image)

        if self.transforms is not None:
            image = self.transforms(image)

        if len(points) == 0:
            points = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            points = torch.tensor(points, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            bboxes = torch.tensor(bboxes, dtype=torch.float32)

        return {
            "image": image,
            "points": points,
            "labels": labels,
            "bboxes": bboxes,
            "image_id": image_id,
        }