import matplotlib.pyplot as plt
import torch

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
    for data in dataloader:
        output = model(data)
        lane_position = data["lane_position"].squeeze(0)
        lane_mask = data["lane_mask"].squeeze(0)
        
        for ith in range(128):
            if lane_mask[ith] == True:
                break
            plt.plot(lane_position[ith, :, 0], lane_position[ith, :, 1], color='black')
            #print(lane_position[ith])
        
        cur_pos = data["global_position"][0, 0, 49]
        cur_head = data["global_heading"][0, 0, 49]
        position = data["global_position"][0]
        mask = data["padding_mask"][0]
        rot_mat = torch.zeros(2, 2)
        rot_mat[0, 0] = cur_head.cos()
        rot_mat[0, 1] = cur_head.sin()
        rot_mat[1, 0] = -cur_head.sin()
        rot_mat[1, 1] = cur_head.cos()

        pred = torch.matmul(output['loc_pos'][0].detach(), rot_mat) + cur_pos
        for ith in range(6):
            plt.plot(pred[ith, :, 0], pred[ith, :, 1], color='green')


        #plt.plot(pred[0, :, 0], pred[0, :, 1], color='green')
        gt = torch.matmul(data['target'][0, 0, :, :2], rot_mat) + cur_pos
        #plt.plot(gt[:, 0], gt[:, 1], color='red')

        for ith in range(40):
            if torch.sum(~mask[ith]) == 0:
                break
            valid_traj = position[ith, ~mask[ith]]
            if ith == 0:
                color = 'red'
            else:
                color = 'blue'
            plt.plot(valid_traj[:, 0], valid_traj[:, 1], color=color)

        
        #ade = torch.norm(pred[0] - gt, dim=-1)
        #print(ade)

    
    plt.savefig('1.jpg')