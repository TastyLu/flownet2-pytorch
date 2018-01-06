import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F

class FlowWarper(nn.Module):
    def __init__(self, w, h):
        super(FlowWarper, self).__init__()
        x = np.arange(0, w)
        y = np.arange(0, h)
        gx, gy = np.meshgrid(x, y, indexing='ij')
        self.w = w
        self.h = h
        self.grid_x = torch.Tensor(gx)
        self.grid_y = torch.Tensor(gy)

    def forward(self, img, uv):
        u = uv[:,0,:,:]
        v = uv[:,1,:,:]
        X = self.grid_x.unsqueeze(0).expand_as(u) + u
        Y = self.grid_y.unsqueeze(0).expand_as(v) + v
        X = 2 * (X/self.w - 0.5).cuda()
        Y = 2 * (Y/self.h - 0.5).cuda()
        grid_tf = torch.stack((X,Y), dim=3)
        img_tf = F.grid_sample(img, grid_tf)
        return img_tf
