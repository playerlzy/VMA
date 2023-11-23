import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import minADE
from metrics import minFDE
from losses import NLLLoss
from losses import MixtureNLLLoss
from models.encoder import VMAEncoder
from models.decoder import SimpleDecoder
from utils.optim import WarmupCosLR

class VMA(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 m2m_layers: int,
                 a2a_layers: int,
                 num_map_types: int,
                 num_mark_types: int,
                 num_is_inter: int,
                 num_lane_edge: int,
                 num_agent_types: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: int,
                 num_modes: int,
                 output_dim: int,
                 output_head: bool,
                 lr: float,
                 warm_up_epoch: int,
                 weight_decay: float,
                 T_max: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim

        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes

        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.warm_up_epoch = warm_up_epoch
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.output_head = output_head
        self.output_dim = output_dim

        self.encoder = VMAEncoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            m2m_layers=m2m_layers,
            a2a_layers=a2a_layers,
            num_map_types=num_map_types,
            num_mark_types=num_mark_types,
            num_is_inter=num_is_inter,
            num_lane_edge=num_lane_edge,
            num_agent_types=num_agent_types,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )

        self.decoder = SimpleDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.minADE = minADE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)

    def forward(self, data):
        x_a = self.encoder(data)
        pred = self.decoder(x_a.reshape(-1, self.hidden_dim))
        return pred

    def training_step(self, data, batch_idx):

        #torch.autograd.set_detect_anomaly(True)

        reg_mask = ~data['padding_mask'][:, :, self.num_historical_steps:].reshape(-1, self.num_future_steps)
        cls_mask = ~data['padding_mask'][:, :, -1].reshape(-1)
        pred = self(data)

        traj = torch.cat([pred['loc_pos'][..., :self.output_dim],
                          pred['loc_head'],
                          pred['scale_pos'][..., :self.output_dim],
                          pred['conc_head']], dim=-1)
        
        pi = pred['pi']
        gt = data['target'].reshape(-1, self.num_future_steps, self.output_dim + self.output_head)

        l2_norm = (torch.norm(traj[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_best = traj[torch.arange(traj.size(0)), best_mode]
        reg_loss = self.reg_loss(traj_best,
                                 gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss = reg_loss.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss = reg_loss.mean()
        cls_loss = self.cls_loss(pred=traj[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss', reg_loss.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss + cls_loss
        return loss

    def validation_step(self, data, batch_idx):

        reg_mask = ~data['padding_mask'][:, :, self.num_historical_steps:].reshape(-1, self.num_future_steps)
        cls_mask = ~data['padding_mask'][:, :, -1].reshape(-1)
        pred = self(data)

        traj = torch.cat([pred['loc_pos'][..., :self.output_dim],
                          pred['loc_head'],
                          pred['scale_pos'][..., :self.output_dim],
                          pred['conc_head']], dim=-1)

        pi = pred['pi']
        gt = data['target'].reshape(-1, self.num_future_steps, self.output_dim + self.output_head)
        
        #torch.set_printoptions(threshold=float('inf'))
        file = open('1.txt', 'w')
        print(traj, file=file)
        print(data, file=file)
        file.close()
        
        l2_norm = (torch.norm(traj[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_best = traj[torch.arange(traj.size(0)), best_mode]
        reg_loss = self.reg_loss(traj_best,
                                 gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss = reg_loss.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss = reg_loss.mean()
        cls_loss = self.cls_loss(pred=traj[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss', reg_loss.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('val_cls_loss', cls_loss.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        
        eval_mask = (data['object_category'] == 3).reshape(-1)

        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj[eval_mask, :, :, :self.output_dim + self.output_head]
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)

        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warm_up_epoch,
            epochs=self.T_max,
        )
        return [optimizer], [scheduler]