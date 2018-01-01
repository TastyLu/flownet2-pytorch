#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils, tools
from networks import SpyNet



parser = argparse.ArgumentParser()
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

args = parser.parse_args()

if torch.cuda.is_available():
    model = torch.nn.DataParallel(SpyNet())
    model.cuda()
