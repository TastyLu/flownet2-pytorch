#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils
import SpyNet
from spynet_transform import FlowWarper
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=4)
parser.add_argument('--number_gpus', '-ng', type=int, default=1, help='number of GPUs to use')

parser.add_argument('-fineWidth', type=int, default=512, help='the length of the fine flow field')
parser.add_argument('-fineHeight', type=int, default=384, help='the width of the fine flow field')
parser.add_argument('-level', type=int, default=1, help='Options: 1,2,3.., whether to initialize flow to zero')

parser.add_argument('-augment', type=int, default=1, help='augment the data')
parser.add_argument('-nEpochs', type=int, default=1000, help='Number of total epochs to run')
parser.add_argument('-epochSize', type=int, default=1000, help='Number of batches per epoch')
parser.add_argument('-batchSize', type=int, default=32, help='mini-batch size')

parser.add_argument('-L1', type=str, default='models/modelL1_4.t7', help='Trained Level 1 model')
parser.add_argument('-L2', type=str, default='models/modelL2_4.t7', help='Trained Level 2 model')
parser.add_argument('-L3', type=str, default='models/modelL3_4.t7', help='Trained Level 3 model')
parser.add_argument('-L4', type=str, default='models/modelL4_4.t7', help='Trained Level 4 model')

# parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
# parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
args = parser.parse_args()

args.crop_size = [384, 512]
args.inference_size = [384, 512]

train_dataset = datasets.MpiSintel(args, is_cropped=True, root='/home/luwei/mpi-sintel/training', replicates=1)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4, pin_memory=True)

validation_dataset = datasets.MpiSintel(args, is_cropped=True, root='/home/luwei/mpi-sintel/training',
                                           replicates=1)
validation_loader = DataLoader(validation_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4,
                               pin_memory=True)

modelL1path = args.L1
modelL2path = args.L2
modelL3path = args.L3
modelL4path = args.L4

global modelL1, modelL2, modelL3, modelL4
global down1, down2, down3, down4

if args.level > 1:
    # Load modelL1
    modelL1 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
    modelL1.load_state_dict(torch.load(modelL1path))
    modelL1.train(False)
    down1 = nn.AvgPool2d(2, stride=2)

if args.level > 2:
    # Load modelL2
    modelL2 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
    modelL2.load_state_dict(torch.load(modelL2path))
    modelL2.train(False)
    down2 = nn.AvgPool2d(2, stride=2)

if args.level > 3:
    # Load modelL3
    modelL3 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
    modelL3.load_state_dict(torch.load(modelL3path))
    modelL3.train(False)
    down3 = nn.AvgPool2d(2, stride=2)

if args.level > 4:
    # Load modelL4
    modelL4 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
    modelL4.load_state_dict(torch.load(modelL4path))
    modelL4.train(False)
    down4 = nn.AvgPool2d(2, stride=2)


def computeInitFlowL1(imagesL1):
    # imageL1 B*6*h*w
    # flow_est B*2*h*w
    h = imagesL1.size(3)
    w = imagesL1.size(4)
    batchSize = imagesL1.size(1)

    _flowappend = torch.cuda.FloatTensor(batchSize, 2, h, w).fill_(0)
    images_in = Variable(torch.cat([imagesL1, _flowappend], 1))

    flow_est = modelL1(images_in)
    return flow_est


def computeInitFlowL2(imagesL2):
    # imageL2 B*6*h*w
    h = imagesL2.size(3)
    w = imagesL2.size(4)
    imagesL1 = down2(imagesL2.clone())
    _flowappend = F.upsample(computeInitFlowL1(imagesL1), scale_factor=2, mode='bilinear')
    warp2 = FlowWarper(h, w)

    _img2 = imagesL2[:, 3:6, :, :]
    imagesL2[:, 3:6, :, :] = warp2(_img2, _flowappend)

    images_in = Variable(torch.cat([imagesL2, _flowappend], 1))
    flow_est = modelL2(images_in)
    return flow_est + _flowappend


def computeInitFlowL3(imagesL3):
    h = imagesL3.size(3)
    w = imagesL3.size(4)
    imagesL2 = down3(imagesL3.clone())
    _flowappend = F.upsample(computeInitFlowL2(imagesL2), scale_factor=2, mode='bilinear')
    warp3 = FlowWarper(h, w)

    _img2 = imagesL3[:, 3:6, :, :]
    imagesL3[:, 3:6, :, :] = warp3(_img2, _flowappend)

    images_in = Variable(torch.cat([imagesL3, _flowappend], 1))
    flow_est = modelL3(images_in)
    return flow_est + _flowappend


def computeInitFlowL4(imagesL4):
    h = imagesL4.size(3)
    w = imagesL4.size(4)
    imagesL3 = down4(imagesL4.clone())
    _flowappend = F.upsample(computeInitFlowL3(imagesL3), scale_factor=2, mode='bilinear')
    warp4 = FlowWarper(h, w)

    _img2 = imagesL4[:, 3:6, :, :]
    imagesL4[:, 3:6, :, :] = warp4(_img2, _flowappend)

    images_in = Variable(torch.cat([imagesL4, _flowappend], 1))
    flow_est = modelL4(images_in)
    return flow_est + _flowappend


def makeData(images, flows):
    # input images 3*2*h*w Tensor
    # input flows 2*h*w
    # output imageFlowInputs 8*h'*w'
    # output flowDiffOutput 2*h'*w'

    # transform to 2*3*h*w
    images = images.permute(1,0,2,3)
    scale1 = transforms.Resize((args.fineHeight / 2 ** 4, args.fineWidth / 2 ** 4))
    trans1 = transforms.Compose([transforms.ToPILImage(), scale1, transforms.ToTensor()])
    scale2 = transforms.Resize((args.fineHeight / 2 ** 3, args.fineWidth / 2 ** 3))
    trans2 = transforms.Compose([transforms.ToPILImage(), scale2, transforms.ToTensor()])
    scale3 = transforms.Resize((args.fineHeight / 2 ** 2, args.fineWidth / 2 ** 2))
    trans3 = transforms.Compose([transforms.ToPILImage(), scale3, transforms.ToTensor()])
    scale4 = transforms.Resize((args.fineHeight / 2 ** 1, args.fineWidth / 2 ** 1))
    trans4 = transforms.Compose([transforms.ToPILImage(), scale4, transforms.ToTensor()])


    if args.level == 1:

        images_scaled = torch.cat([trans1(images[0]), trans1(images[1])], 0) # 6*h*w
        initFlow = torch.zeros(1, 2, args.fineHeight / 2 ** 4, args.fineWidth / 2 ** 4)
        flowDiffOutput = F.avg_pool2d(flows, 2**4) # 2*h*w

    elif args.level == 2:

        images_scaled = torch.cat([trans2(images[0]), trans2(images[1])], 0)

        imageL1 = torch.cat([trans1(images[0]), trans1(images[1])], 0)
        initFlow = computeInitFlowL1(torch.unsqueeze(imageL1, 0).cuda())
        initFlow = F.upsample(initFlow, scale_factor=2, mode='bilinear') # B*2*h*w
        flowDiffOutput = torch.unsqueeze(torch.stack([scale2(flows[0]), scale2(flows[1])]), 0) - initFlow #B*2*h*w

    elif args.level == 3:
        images_scaled = torch.cat([trans3(images[0]), trans3(images[1])], 0)

        imageL2 = torch.cat([trans2(images[0]), trans2(images[1])], 0)
        initFlow = computeInitFlowL2(torch.unsqueeze(imageL2, 0).cuda())
        initFlow = F.upsample(initFlow, scale_factor=2, mode='bilinear')
        flowDiffOutput = torch.unsqueeze(torch.stack([scale3(flows[0]), scale3(flows[1])]), 0) - initFlow

    elif args.level == 4:
        images_scaled = torch.cat([trans4(images[0]), trans4(images[1])], 0)

        imageL3 = torch.cat([trans3(images[0]), trans3(images[1])], 0)
        initFlow = computeInitFlowL3(torch.unsqueeze(imageL3, 0).cuda())
        initFlow = F.upsample(initFlow, scale_factor=2, mode='bilinear')
        flowDiffOutput = torch.unsqueeze(torch.stack([scale4(flows[0]), scale4(flows[1])]), 0) - initFlow

    elif args.level == 5:
        images_scaled = torch.cat([images[0], images[1]], 0)

        imageL4 = torch.cat([trans4(images[0]), trans4(images[1])], 0)
        initFlow = computeInitFlowL4(torch.unsqueeze(imageL4, 0).cuda())
        initFlow = F.upsample(initFlow, scale_factor=2, mode='bilinear')
        flowDiffOutput = torch.unsqueeze(torch.stack([flows[0], flows[1]]), 0) - initFlow

    _img2 = images_scaled[3:6, :, :].clone()
    _img2 = torch.unsqueeze(_img2, 0).cuda()
    #print initFlow.size(), _img2.size()
    warper = FlowWarper(initFlow.size(2), initFlow.size(3))
    images_scaled[3:6, :, :] = torch.squeeze(warper(_img2, initFlow), 0).data
    imageFlowInputs = torch.cat([images_scaled, torch.squeeze(initFlow, 0)], 0)
    return imageFlowInputs, torch.squeeze(flowDiffOutput, 0)


if torch.cuda.is_available():
    spyNet = SpyNet.SpyNet(args)
    model = nn.parallel.DataParallel(spyNet, device_ids=list(range(args.number_gpus)))
    model.cuda()
    torch.cuda.manual_seed(args.seed)

epochs = 201
lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
epeLoss = losses.L1Loss(args)
#model.load_state_dict(torch.load("./pth_fine/fcn-deconv-100.pth"))
model.train()

for epoch in range(epochs):
    running_l1_loss = 0.0
    running_epe_loss = 0.0
    iter_l1_loss = 0.0
    iter_epe_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
	data = data[0]
        target = target[0]
        imageFlowList = []
        flowDiffList = []
	batchSize = data.size(0)
        for idx in range(batchSize):
            # should do multi thread
            image, flow = makeData(data[idx], target[idx])
            imageFlowList.append(image)
            flowDiffList.append(flow)

        imageFlowInputs = Variable(torch.stack(imageFlowList).cuda())
        flowDiffOutputs = torch.stack(flowDiffList).cuda()

        optimizer.zero_grad()
        outputs = model(imageFlowInputs)
        loss = epeLoss(outputs, flowDiffOutputs)
	#print loss
        #loss.backward()
        optimizer.step()
        running_l1_loss += loss[0].data[0]
        running_epe_loss += loss[1].data[0]
        iter_l1_loss += loss[0].data[0]
        iter_epe_loss += loss[1].data[0]
        if (batch_idx + 1) % 100 == 0:
            print("Iter [%d] L1Loss: %.4f EPELoss: %.4f" % (batch_idx + 1, iter_l1_loss / 100.0, iter_epe_loss / 100.0))
            iter_l1_loss = 0.0
            iter_epe_loss = 0.0
    print("Epoch [%d] L1Loss: %.4f EPELoss: %.4f" % (epoch + 1, running_l1_loss / batch_idx, running_epe_loss / 100.0))
    running_l1_loss = 0
    running_epe_loss = 0

    if (epoch + 1) % 50 == 0:
        lr /= 10.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        torch.save(model.state_dict(), "./pth/spynet-L%d-%d.pth" % (args.level, epoch + 1))

torch.save(model.state_dict(), "./pth/spynet-L%d.pth" % (args.level))
