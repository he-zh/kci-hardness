import torch
import torch.nn as nn
from utils import add_diag


class LeaveOneOutCriterion(nn.Module):

    def forward(self, K_yy, K_QQ, reg):
        n = K_yy.shape[0]
        K_yy = K_yy.double()
        K_QQ = K_QQ.double()
        reg = reg.double()
        Kinv = torch.linalg.solve(add_diag(K_yy, n * reg), K_yy).T

        value = ((K_QQ.diagonal() + (Kinv @ K_QQ @ Kinv.T).diagonal() -
                2 * (Kinv @ K_QQ).diagonal()) / (1 - Kinv.diagonal()) ** 2).mean()
        return value.float()


