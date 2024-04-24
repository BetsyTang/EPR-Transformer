import torch
import torch.nn as nn
from .sdtw_cuda_loss import SoftDTW

class ExpressionLoss(nn.Module):
    def __init__(self, output_features, output_feature_boundaries, normalize=False):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.soft_dtw = SoftDTW(use_cuda=True, gamma=0.1)

        #setting these to 0.25 so their total starts out as 1.0. 
        self.out_feat = output_features
        self.out_feat_bound = output_feature_boundaries
        self.normalize = normalize

    def forward(self, x, y):
        losses = {}
        for i in range(len(self.out_feat)):
            target = y[:,:,i].unsqueeze(-1)
            predict = x[i] #make the predictions as integers
            if self.normalize:
                l1_loss = self.l1_loss(predict, target) / self.out_feat_bound[i][1]
            else:
                l1_loss = self.l1_loss(predict, target) 
                
            # soft_dtw_loss = self.soft_dtw(predict, target)
            # soft_dtw_loss_value = soft_dtw_loss.clone()

            losses[f"{self.out_feat[i]}"] = {
                "l1_loss": l1_loss,
            }
        
        return losses 

class ExpressionDWTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_dtw = SoftDTW(use_cuda=True, gamma=0.1)

    def forward(self, x, y):
        soft_dtw_loss = self.soft_dtw(torch.stack(x, dim=-1), y)
        return soft_dtw_loss