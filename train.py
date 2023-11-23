import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import yaml

from datamodule import Av2Module
from VMA_pred import VMA

if __name__ == '__main__':
    pl.seed_everything(3407, workers=True)

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model = VMA(**config["predict"]["model"])
    datamodule = Av2Module(**config["predict"]["data"])
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=config["predict"]["accelerator"], devices=config["predict"]["device"],
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=config["predict"]["max_epoch"],
                         gradient_clip_val=config["predict"]["gradient_clip_val"],
                         gradient_clip_algorithm=config["predict"]["gradient_clip_algorithm"])
    trainer.fit(model, datamodule)