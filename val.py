import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import yaml

from datamodule import Av2Module
from VMA_pred import VMA

if __name__ == '__main__':
    #pl.seed_everything(3407, workers=True)

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model = VMA(**config["predict"]["model"]).load_from_checkpoint(checkpoint_path='1.ckpt')
    datamodule = Av2Module(**config["predict"]["data"])
    datamodule.setup(1)
    dataloader = datamodule.val_dataloader()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=config["predict"]["accelerator"], devices=config["predict"]["device"],
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True))
    trainer.validate(model, dataloader)