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
from skimage import io, transform

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

#parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
#parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
args = parser.parse_args()

args.crop_size = [384, 512]
args.inference_size = [384, 512]

train_dataset = datasets.FlyingChairs(args, is_cropped=True, root = '/path/to/FlyingChairs_release/data', replicates = 1)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4, pin_memory=True)

validation_dataset = datasets.FlyingChairs(args, is_cropped=True, root = '/path/to/FlyingChairs_release/data', replicates = 1)
validation_loader = DataLoader(validation_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4, pin_memory=True)

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

if args.level > 2 :
    # Load modelL2
   modelL2 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
   modelL2.load_state_dict(torch.load(modelL2path))
   modelL2.train(False)
   down2 = nn.AvgPool2d(2, stride=2)

if args.level > 3 :
   # Load modelL3
   modelL3 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
   modelL3.load_state_dict(torch.load(modelL3path))
   modelL3.train(False)
   down3 = nn.AvgPool2d(2, stride=2)

if args.level > 4 :
   # Load modelL4
   modelL4 = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
   modelL4.load_state_dict(torch.load(modelL4path))
   modelL4.train(False)
   down4 = nn.AvgPool2d(2, stride=2)


def scaleFlow(flow, height, width):
    #scale the original flow (u,v) to ((x+u)/w,(y+v)/h)
    #input flow B*2*u*v
    coord = torch.FloatTensor()

    for id0, item0 in enumerate(flow):
        for id1, item1 in enumerate(flow[id0]):
            for id2, item2 in enumerate(flow[id0][id1]):
                flow[id0][id1]

    sc = height/flow.size(2)
    flow_scaled = Image.scale(flow, width, height)*sc
    return flow_scaled


def computeInitFlowL1(imagesL1):
    h = imagesL1.size(3)
    w = imagesL1.size(4)
    batchSize = imagesL1.size(1)

    _flowappend = torch.cuda.FloatTensor(batchSize, 2, h, w).fill_(0)
    images_in = Variable(torch.cat([imagesL1, _flowappend], 1))

    flow_est = modelL1(images_in)
    return flow_est

def computeInitFlowL2(imagesL2):
    h = imagesL2.size(3)
    w = imagesL2.size(4)
    imagesL1 = down2(imagesL2.clone())
    _flowappend = F.upsample(computeInitFlowL1(imagesL1), scale_factor=2, mode='bilinear')
    _flowappend = scaleFlow(_flowappend, h, w)

    _img2 = imagesL2[:, 3:6, :, :]
    imagesL2[:, 3:6, :, :] = F.grid_sample(_img2, _flowappend, mode='bilinear')

    images_in = Variable(torch.cat([imagesL2, _flowappend], 1))
    flow_est = modelL2(images_in)
    return flow_est + _flowappend

def computeInitFlowL3(imagesL3):
    h = imagesL3.size(3)
    w = imagesL3.size(4)
    imagesL2 = down3(imagesL3.clone())
    _flowappend = F.upsample(computeInitFlowL2(imagesL2), scale_factor=2, mode='bilinear')
    _flowappend = scaleFlow(_flowappend, h, w)

    _img2 = imagesL3[:, 3:6, :, :]
    imagesL3[:, 3:6, :, :] = F.grid_sample(_img2, _flowappend, mode='bilinear')

    images_in = Variable(torch.cat([imagesL3, _flowappend], 1))
    flow_est = modelL3(images_in)
    return flow_est + _flowappend

def computeInitFlowL4(imagesL4):
    h = imagesL4.size(3)
    w = imagesL4.size(4)
    imagesL3 = down4(imagesL4.clone())
    _flowappend = F.upsample(computeInitFlowL3(imagesL3), scale_factor=2, mode='bilinear')
    _flowappend = scaleFlow(_flowappend, h, w)

    _img2 = imagesL4[:, 3:6, :, :]
    imagesL4[:, 3:6, :, :] = F.grid_sample(_img2, _flowappend, mode='bilinear')

    images_in = Variable(torch.cat([imagesL4, _flowappend], 1))
    flow_est = modelL4(images_in)
    return flow_est + _flowappend


def makeData(images, flows):
    # input: 2 images 1 flow
    # images: numpy ndarray


    if args.level == 1:
        initFlow = torch.zeros(2, args.fineHeight, args.fineWidth)
        flowDiffOutput = flows

    elif args.level == 2:
        coarseImages = transform.scale(images, 0.5)
        initFlow = computeInitFlowL1(coarseImages.resize(1, coarseImages.size(1), coarseImages.size(2), coarseImages.size(3)).cuda())
        initFlow = scaleFlow(initFlow.squeeze().float(), args.fineHeight, args.fineWidth)

        flowDiffOutput = scaleFlow(flows, args.fineHeight, args.fineWidth)
        flowDiffOutput = flowDiffOutput.add(flowDiffOutput, -1, initFlow)

    elif args.level == 3:
        coarseImages = transform.scale(images, 0.5)
        initFlow = computeInitFlowL2(coarseImages.resize(1, coarseImages.size(1),
                                                                         coarseImages.size(2), coarseImages.size(3)).cuda())
        initFlow = scaleFlow(initFlow.squeeze().float(), args.fineHeight, args.fineWidth)
        
        flowDiffOutput = scaleFlow(flows, args.fineHeight, args.fineWidth)
        flowDiffOutput = flowDiffOutput.add(flowDiffOutput, -1, initFlow)

    elif args.level == 4:
        coarseImages = transform.scale(images, args.fineWidth / 2, args.fineHeight / 2)
        initFlow = computeInitFlowL3(coarseImages.resize(1, coarseImages.size(1),
                                                                         coarseImages.size(2), coarseImages.size(3)).cuda())
        initFlow = scaleFlow(initFlow.squeeze().float(), args.fineHeight, args.fineWidth)
        
        flowDiffOutput = scaleFlow(flows, args.fineHeight, args.fineWidth)
        flowDiffOutput = flowDiffOutput.add(flowDiffOutput, -1, initFlow)
        
    elif args.level == 5:
        coarseImages = transform.scale(images, args.fineWidth / 2, args.fineHeight / 2)
        initFlow = computeInitFlowL4(coarseImages.resize(1, coarseImages.size(1),
                                                                         coarseImages.size(2), coarseImages.size(3)).cuda())
        initFlow = scaleFlow(initFlow.squeeze().float(), args.fineHeight, args.fineWidth)
        
        flowDiffOutput = scaleFlow(flows, args.fineHeight, args.fineWidth)
        flowDiffOutput = flowDiffOutput.add(flowDiffOutput, -1, initFlow)

    _img2 = images_scaled[{{4, 6}, {}, {}}].clone()
    images_scaled[{{4, 6}, {}, {}}].copy(image.warp(_img2, initFlow.index(1, torch.LongTensor{2, 1})))
    imageFlowInputs = torch.cat(images_scaled, initFlow.float(), 1)
    return imageFlowInputs, flowDiffOutput


if torch.cuda.is_available():
    model = torch.nn.DataParallel(SpyNet(), device_ids=list(range(args.number_gpus)))
    model.cuda()
    torch.cuda.manual_seed(args.seed)

epochs = 201
lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
epeLoss = losses.L1Loss()
model.load_state_dict(torch.load("./pth_fine/fcn-deconv-100.pth"))
model.train()



for epoch in range(epochs):
    running_loss = 0.0
    iter_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        imageFlowInputs, flowDiffOutput = makeData(data, target)

        if torch.cuda.is_available():
            data, target = [Variable(d.cuda()) for d in data], [Variable(t.cuda()) for t in target]
        else:
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]


        optimizer.zero_grad()
        outputs = model(data[0])
        loss = epeLoss(outputs, target[0])
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        iter_loss += loss.data[0]
        if (batch_idx + 1) % 100 == 0:
            print("Iter [%d] Loss: %.4f" % (batch_idx+1, iter_loss/100.0))
            iter_loss = 0.0

    print("Epoch [%d] Loss: %.4f" % (epoch+1, running_loss/batch_idx))
    running_loss = 0

    if (epoch+1) % 50 == 0:
        lr /= 10.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        torch.save(model.state_dict(), "./pth/fcn-deconv-%d.pth" % (epoch+1))

torch.save(model.state_dict(), "./pth/fcn-deconv.pth")
