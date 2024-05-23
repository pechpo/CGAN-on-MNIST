import torch
from torch import nn
from torch.autograd import Variable
import random

import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision.datasets import MNIST

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEED = 1642
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

NOISE_DIM = 96
NUM_EPOCHS = 10
batch_size = 128

def show_images(): # 定义画图工具
    return 

# 归一化
def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    return (x + 1.0) / 2.0

#训练集
train_set = Dataset()

train_data = DataLoader()

# 判别器
def discriminator():
    net = nn.Sequential(        

        )
    return net

#生成器
def generator(noise_dim=NOISE_DIM):   
    net = nn.Sequential(

    )
    return net

bce_loss = nn.BCEWithLogitsLoss()

# 判别器损失
def discriminator_loss():
    return

# 生成器损失
def generator_loss():
    return

# 优化器
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

# 训练过程
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, noise_size=NOISE_DIM, num_epochs=NUM_EPOCHS):
    return

D = discriminator().cuda()
G = generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)