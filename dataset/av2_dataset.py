from pathlib import Path

import torch
from torch.utils.data import Dataset

import numpy as np

class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        transform: object,
        cached_split: str = None
    ):
        super(Av2Dataset, self).__init__()

        if cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(list(self.data_folder.glob("*.pt")))
            self.data = []
            self.num_data = len(self.file_list)
            self.load = True

        self.transform = transform

        print(
            f"data root: {data_root}/{cached_split}, total number of files: {self.num_data}"
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        #index = np.random.randint(self.num_data)
        #print(index)
        index = 12422
        data = torch.load(self.file_list[index])
        print(index, data["scenario_id"])
        return self.transform(data)