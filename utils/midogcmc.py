


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





midogcmc = MidogCMC()
