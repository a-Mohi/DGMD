


import sqlite3
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict



class MidogCMC(Dataset):
    def __init__(self, root_dir, sqlite_path, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.sqlite_path = sqlite_path
        self.source = "CMC"
        self.conn = sqlite3.connect(self.sqlite_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("SELECT uid, filename FROM Slides;")
        slides_data = self.cursor.fetchall()

        # slides_uid -> full image path
        self.slides = {
            uid: f"{self.root_dir}/{self.filename}" for uid, filename in slides_data
        }

        self.cursor.execute("SELECT uid, name from Classes;")
        self.classes = dict(self.cursor.fetchall())

        self.cursor.execute("""
            SELECT A.uid, A.slide, L.class
            FROM Annotations A
            JOIN Annotations_label L 
            ON A.uid = L.annoId
        """)

        ann_data = self.cursor.fetchall()
        self.cursor.execute("""
            SELECT annoId, coordinateX, coordinateY
            FROM Annotations_coordinates
            ORDER BY annoId, orderIdx
        """)

        coord_data = self.cursor.fetchall()

        coords_by_ann = defaultdict(list)

        for annoId, x, y in coord_data:
            self.coords_by_ann[annoId] = (x, y)
        
        self.samples = defaultdict(list)

        for anno_id, slide_uid, class_id in ann_data:
            if anno_id not in coords_by_ann:
                continue
            sample = {
                "label": class_id,
                "class_name": self.classes.get(class_id, None),
                "points": coords_by_ann[anno_id]
            }

            if slide_uid in self.samples:
                self.samples[slide_uid]["instances"].append(sample)
            else:
                self.samples[slide_uid] = {
                    "image_id": slide_uid,
                    "slide_path": self.slides[slide_uid],
                    "instances": [sample]
                }
            
            print(f"Loaded {len(self.ann_data)} annotations.")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            image_data = self.samples[idx]
            return image_data

