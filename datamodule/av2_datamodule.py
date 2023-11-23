from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from transforms import NaiveTransform
from dataset import Av2Dataset


class Av2Module(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        num_historical_steps: int,
        num_future_steps: int,
        a2m_radius: int,
        a2a_radius: int,
        m2m_radius: int,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.a2a_radius = a2a_radius
        self.a2m_radius = a2m_radius
        self.m2m_radius = m2m_radius

    def setup(self, stage) -> None:
        self.train_dataset = Av2Dataset(
            data_root=self.data_root, 
            transform= NaiveTransform(self.num_historical_steps, 
                                      self.num_future_steps, 
                                      self.a2m_radius,
                                      self.a2a_radius,
                                      self.m2m_radius), 
            cached_split="train"
        )
        self.val_dataset = Av2Dataset(
            data_root=self.data_root, 
            transform= NaiveTransform(self.num_historical_steps, 
                                      self.num_future_steps, 
                                      self.a2m_radius,
                                      self.a2a_radius,
                                      self.m2m_radius), 
            cached_split="val"
        )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
