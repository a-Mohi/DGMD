import csv
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from pathlib import Path
import torch

class MiDeSec(Dataset):
    """
    MiDeSec Dataset loading dataset

    Classes:
        1 -> mitotic figure
        background is implicit
    """

    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir (str | Path): directory containing images
            transforms (callable, optional): image transforms only
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.source = "midesec"
        grouped = defaultdict(dict)

        # Iterate through all files in root_dir
        for file_path in sorted(self.root_dir.iterdir()):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            stem = file_path.stem

            if ext == ".pt":
                grouped[stem]["pt"] = str(file_path)
            elif ext == ".csv":
                grouped[stem]["csv"] = str(file_path)

        # Convert to required list of dicts
        self.data = []
        for stem, files in grouped.items():
            if "pt" in files and "csv" in files:
                self.data.append({
                    "id": stem,
                    "pt": files["pt"],
                    "csv": files["csv"],
                })

    @staticmethod
    def polygon_to_bbox(coords):
        xs = coords[0::2]
        ys = coords[1::2]

        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)

        return xmin, ymin, xmax, ymax

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_info = self.data[idx]
        image_id = img_info['id']
        img_path = img_info['pt']
        image = torch.load(img_path, map_location="cpu", mmap=True)

        ann_file = img_info['csv']

        points = []
        labels = []
        bboxes = []
        category_id = 1


        with open(ann_file, newline="") as infile:
            reader = csv.reader(infile)

            for row in reader:
                coords = list(map(float, row))
                x1, y1, x2, y2 = self.polygon_to_bbox(coords)

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