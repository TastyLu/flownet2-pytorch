import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from correlation_package.modules.correlation import Correlation
from networks.resample2d_package.modules.resample2d import Resample2d

from submodules import *

class PWCNet(nn.Module):
    def __init__(self, args, input_channels = 3, batchNorm=True):
        # Feature Pyramid Extractor Network
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,  input_channels,   16, kernel_size=3, stride=2)
        self.conv1_1   = conv(self.batchNorm,  16,  16, kernel_size=3) # c_t^1 shape: 16X256X256
        self.conv2   = conv(self.batchNorm,  16,  32, kernel_size=3, stride=2)
        self.conv2_1 = conv(self.batchNorm,  32,  32, kernel_size=3) # c_t^2 shape: 32X128X128
        self.conv3   = conv(self.batchNorm,  32,  64, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batchNorm,  64,  64, kernel_size=3) # c_t^3 shape: 64X64X64
        self.conv4   = conv(self.batchNorm,  64,  96, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batchNorm,  96,  96, kernel_size=3) # c_t^4 shape: 96X32X32
        self.conv5   = conv(self.batchNorm,  96, 128, kernel_size=3, stride=2)
        self.conv5_1 = conv(self.batchNorm, 128, 128, kernel_size=3) # c_t^5 shape: 128X16X16
        self.conv6   = conv(self.batchNorm, 128, 192, kernel_size=3, stride=2)
        self.conv6_1 = conv(self.batchNorm, 192, 192, kernel_size=3) # c_t^6 shape: 192X8X8


        # Optical Flow Estimator network for level-2, feature shape: 32X128X128
        self.flow_level2_corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.flow_level2_corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.flow_level2_conv1   = conv(self.batchNorm, 115, 128, kernel_size=3)
        self.flow_level2_conv2   = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.flow_level2_conv3   = conv(self.batchNorm, 128,  96, kernel_size=3)
        self.flow_level2_conv4   = conv(self.batchNorm,  96,  64, kernel_size=3)
        self.flow_level2_conv5   = conv(self.batchNorm,  64,  32, kernel_size=3)
        self.flow_level2_conv6   = conv(self.batchNorm,  32,   2, kernel_size=3)

        # Optical Flow Estimator network for level-3, feature shape: 64X64X64
        self.flow_level3_corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.flow_level3_corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.flow_level3_conv1   = conv(self.batchNorm, 115, 128, kernel_size=3)
        self.flow_level3_conv2   = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.flow_level3_conv3   = conv(self.batchNorm, 128,  96, kernel_size=3)
        self.flow_level3_conv4   = conv(self.batchNorm,  96,  64, kernel_size=3)
        self.flow_level3_conv5   = conv(self.batchNorm,  64,  32, kernel_size=3)
        self.flow_level3_conv6   = conv(self.batchNorm,  32,   2, kernel_size=3)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Optical Flow Estimator network for level-4, feature shape: 96X32X32
        self.flow_level4_corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.flow_level4_corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.flow_level4_conv1   = conv(self.batchNorm, 115, 128, kernel_size=3)
        self.flow_level4_conv2   = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.flow_level4_conv3   = conv(self.batchNorm, 128,  96, kernel_size=3)
        self.flow_level4_conv4   = conv(self.batchNorm,  96,  64, kernel_size=3)
        self.flow_level4_conv5   = conv(self.batchNorm,  64,  32, kernel_size=3)
        self.flow_level4_conv6   = conv(self.batchNorm,  32,   2, kernel_size=3)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Optical Flow Estimator network for level-5, feature shape: 128X16X16
        self.flow_level5_corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.flow_level5_corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.flow_level5_conv1   = conv(self.batchNorm, 115, 128, kernel_size=3)
        self.flow_level5_conv2   = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.flow_level5_conv3   = conv(self.batchNorm, 128,  96, kernel_size=3)
        self.flow_level5_conv4   = conv(self.batchNorm,  96,  64, kernel_size=3)
        self.flow_level5_conv5   = conv(self.batchNorm,  64,  32, kernel_size=3)
        self.flow_level5_conv6   = conv(self.batchNorm,  32,   2, kernel_size=3)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Optical Flow Estimator network for level-6, feature shape: 192X8X8
        self.flow_level6_corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.flow_level6_corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.flow_level6_conv1   = conv(self.batchNorm, 115, 128, kernel_size=3)
        self.flow_level6_conv2   = conv(self.batchNorm, 128, 128, kernel_size=3)
        self.flow_level6_conv3   = conv(self.batchNorm, 128,  96, kernel_size=3)
        self.flow_level6_conv4   = conv(self.batchNorm,  96,  64, kernel_size=3)
        self.flow_level6_conv5   = conv(self.batchNorm,  64,  32, kernel_size=3)
        self.flow_level6_conv6   = conv(self.batchNorm,  32,   2, kernel_size=3)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Context Network for level-2, input = concat(feature_2, flow)
        self.context_conv1   = conv(self.batchNorm,  34, 128, kernel_size=3, dilation=1)
        self.context_conv2   = conv(self.batchNorm, 128, 128, kernel_size=3, dilation=2)
        self.context_conv3   = conv(self.batchNorm, 128, 128, kernel_size=3, dilation=4)
        self.context_conv4   = conv(self.batchNorm, 128,  96, kernel_size=3, dilation=8)
        self.context_conv5   = conv(self.batchNorm,  96,  64, kernel_size=3, dilation=16)
        self.context_conv6   = conv(self.batchNorm,  64,  32, kernel_size=3, dilation=1)
        self.context_conv7   = conv(self.batchNorm,  32,   2, kernel_size=3, dilation=1)

        # Shared operations
        # self.flow_corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.resample = Resample2d()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:,0:3,:,:]
        x2 = x[:,3::,:,:]

        out_conv1a = self.conv1_1(self.conv1(x1))
        out_conv2a = self.conv2_1(self.conv2(out_conv1a))
        out_conv3a = self.conv3_1(self.conv3(out_conv2a))
        out_conv4a = self.conv4_1(self.conv4(out_conv3a))
        out_conv5a = self.conv5_1(self.conv5(out_conv4a))
        out_conv6a = self.conv6_1(self.conv6(out_conv5a))

        out_conv1b = self.conv1_1(self.conv1(x2))
        out_conv2b = self.conv2_1(self.conv2(out_conv1b))
        out_conv3b = self.conv3_1(self.conv3(out_conv2b))
        out_conv4b = self.conv4_1(self.conv4(out_conv3b))
        out_conv5b = self.conv5_1(self.conv5(out_conv4b))
        out_conv6b = self.conv6_1(self.conv6(out_conv5b))

        out_corr_6 = self.flow_level6_corr(out_conv6a, out_conv6b)
        out_corr_6 = self.flow_level6_corr_activation(out_corr_6)

        concat6 = torch.cat((out_conv6a, out_corr_6), dim=1)
        flow_level6_conv1 = self.flow_level6_conv1(concat6)
        flow_level6_conv2 = self.flow_level6_conv2(flow_level6_conv1)
        flow_level6_conv3 = self.flow_level6_conv3(flow_level6_conv2)
        flow_level6_conv4 = self.flow_level6_conv4(flow_level6_conv3)
        flow_level6_conv5 = self.flow_level6_conv5(flow_level6_conv4)
        flow_level6 = self.flow_level6_conv6(flow_level6_conv5)

        up_flow_level6 = self.upsample6(flow_level6)
        out_conv5b_warp = self.resample2(out_conv5b, up_flow_level6)
        out_corr_5 = self.flow_level5_corr(out_conv5a, out_conv5b_warp)
        out_corr_5 = self.flow_level5_corr_activation(out_corr_5)
        concat5 = torch.cat((out_conv5a, out_corr_5, up_flow_level6), dim=1)
        flow_level5_conv1 = self.flow_level5_conv1(concat5)
        flow_level5_conv2 = self.flow_level5_conv2(flow_level5_conv1)
        flow_level5_conv3 = self.flow_level5_conv3(flow_level5_conv2)
        flow_level5_conv4 = self.flow_level5_conv4(flow_level5_conv3)
        flow_level5_conv5 = self.flow_level5_conv5(flow_level5_conv4)
        flow_level5 = self.flow_level5_conv6(flow_level5_conv5)

        up_flow_level5 = self.upsample5(flow_level5)
        out_conv4b_warp = self.resample2(out_conv4b, up_flow_level5)
        out_corr_4 = self.flow_level4_corr(out_conv4a, out_conv4b_warp)
        out_corr_4 = self.flow_level4_corr_activation(out_corr_4)
        concat4 = torch.cat((out_conv4a, out_corr_4, up_flow_level5), dim=1)
        flow_level4_conv1 = self.flow_level4_conv1(concat4)
        flow_level4_conv2 = self.flow_level4_conv2(flow_level4_conv1)
        flow_level4_conv3 = self.flow_level4_conv3(flow_level4_conv2)
        flow_level4_conv4 = self.flow_level4_conv4(flow_level4_conv3)
        flow_level4_conv5 = self.flow_level4_conv5(flow_level4_conv4)
        flow_level4 = self.flow_level4_conv6(flow_level4_conv5)

        up_flow_level4 = self.upsample4(flow_level4)
        out_conv3b_warp = self.resample2(out_conv3b, up_flow_level4)
        out_corr_3 = self.flow_level3_corr(out_conv3a, out_conv3b_warp)
        out_corr_3 = self.flow_level3_corr_activation(out_corr_3)
        concat3 = torch.cat((out_conv3a, out_corr_3, up_flow_level4), dim=1)
        flow_level3_conv1 = self.flow_level3_conv1(concat3)
        flow_level3_conv2 = self.flow_level3_conv2(flow_level3_conv1)
        flow_level3_conv3 = self.flow_level3_conv3(flow_level3_conv2)
        flow_level3_conv4 = self.flow_level3_conv4(flow_level3_conv3)
        flow_level3_conv5 = self.flow_level3_conv5(flow_level3_conv4)
        flow_level3 = self.flow_level3_conv6(flow_level3_conv5)

        up_flow_level3 = self.upsample3(flow_level3)
        out_conv2_warp = self.resample2(out_conv2b, up_flow_level3)
        out_corr_2 = self.flow_level2_corr(out_conv2a, out_conv2b_warp)
        out_corr_2 = self.flow_level2_corr_activation(out_corr_2)
        concat2 = torch.cat((out_conv2a, out_corr_2, up_flow_level3), dim=1)
        flow_level2_conv1 = self.flow_level2_conv1(concat2)
        flow_level2_conv2 = self.flow_level2_conv2(flow_level2_conv1)
        flow_level2_conv3 = self.flow_level2_conv3(flow_level2_conv2)
        flow_level2_conv4 = self.flow_level2_conv4(flow_level2_conv3)
        flow_level2_conv5 = self.flow_level2_conv5(flow_level2_conv4)
        flow_level2 = self.flow_level2_conv6(flow_level2_conv5)

        context_concat2 = torch.cat((flow_level2_conv5, flow_level2), dim=1)
        context_conv1 = self.context_conv1(context_concat2)
        context_conv2 = self.context_conv2(context_conv1)
        context_conv3 = self.context_conv3(context_conv2)
        context_conv4 = self.context_conv4(context_conv3)
        context_conv5 = self.context_conv5(context_conv4)
        context_conv6 = self.context_conv6(context_conv5)
        flow_context = self.context_conv7(context_conv6)

        flow_sum = torch.add(flow_level2, flow_context)

        if self.training:
            return flow_sum,flow_context,flow_level2,flow_level3,flow_level4,flow_level5，flow_level6
        else:
            return flow_sum
