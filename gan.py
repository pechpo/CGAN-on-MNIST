import torch
from torch import nn
import random
import cv2

import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
from torch.autograd import Variable
import gradio as gr
import os
os.environ['GRADIO_TEMP_DIR'] = './tmp'

import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

SEED = 1642
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

NOISE_DIM = 64
NUM_EPOCHS = 50
batch_size = 128

# 归一化
def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    return (x + 1.0) / 2.0

#训练集
train_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)
train_data = DataLoader(train_set, batch_size=batch_size)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, 3),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        self.fc_y = nn.Linear(1, 1024)
        
    def forward(self, x, y):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        y = self.fc_y(y)
        x = x + y
        x = self.fc(x)
        return x

#生成器
class Generator(nn.Module): 
    def __init__(self, noise_dim=NOISE_DIM):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 7 * 7 * 256),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 256)
        )
        self.fc_y = nn.Linear(1, 1024)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Tanh()
        )
        
    def forward(self, x, y):
        x = self.fc1(x)
        y = self.fc_y(y)
        x = x + y
        x = self.fc2(x)
        x = x.view(x.shape[0], 256, 7, 7)
        x = self.conv(x)
        return x

bce_loss = nn.BCEWithLogitsLoss()
def discriminator_loss(logits_real, logits_fake): # 判别器的 loss
    size = logits_real.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float().cuda()
    false_labels = Variable(torch.zeros(size, 1)).float().cuda()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss

def generator_loss(logits_fake): # 生成器的 loss  
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float().cuda()
    loss = bce_loss(logits_fake, true_labels)
    return loss

# 优化器
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

# 训练过程
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, noise_size=NOISE_DIM, num_epochs=NUM_EPOCHS):
    D.load_state_dict(torch.load("D_net_param.pth"))
    G.load_state_dict(torch.load("G_net_param.pth"))
    d_error_L = []
    g_error_L = []
    for epoch in range(num_epochs):
        print(f"Training on epoch{epoch}:")
        with tqdm(total=train_data.__len__()) as pbar:
            d_error_l = []
            g_error_l = []
            for x, y in train_data:
                bs = x.shape[0]
                # 判别网络
                real_data = x.cuda()
                label = y.view(bs, 1).float().cuda()
                logits_real = D_net(real_data, label)

                sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
                g_fake_seed = sample_noise.cuda()
                fake_images = G_net(g_fake_seed, label)
                logits_fake = D_net(fake_images, label)

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_error_l.append(d_total_error.item())
                D_optimizer.zero_grad()
                d_total_error.backward()
                D_optimizer.step()
                
                # 生成网络
                g_fake_seed = sample_noise.cuda()
                fake_images = G_net(g_fake_seed, label)

                gen_logits_fake = D_net(fake_images, label)
                g_error = generator_loss(gen_logits_fake)
                G_optimizer.zero_grad()
                g_error.backward()
                G_optimizer.step()

                g_error_l.append(g_error.item())
                pbar.set_postfix({'d_loss':f"{d_total_error.item():.6f}", 'g_loss':f"{g_error.item():.6f}"})
                pbar.update(1)

            d_error = sum(d_error_l) / len(d_error_l)
            d_error_L.append(d_error)
            g_error = sum(g_error_l) / len(g_error_l)
            g_error_L.append(g_error)
            print(f"Discriminating network error:{d_error}") 
            print(f"Generating network error:{g_error}") 

    torch.save(D_net.state_dict(), "D_net_param.pth")
    torch.save(G_net.state_dict(), "G_net_param.pth")
    return d_error_L, g_error_L

def predict(num):
    G.eval()
    with torch.no_grad():
        bs = 16
        noise_size = NOISE_DIM
        label = torch.zeros(bs)
        label += num
        label = label.view(bs, 1).float().cuda()
        sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
        g_fake_seed = sample_noise.cuda()
        fake_images = G(g_fake_seed, label)
        image = fake_images.detach().cpu().numpy()
        image = image.transpose((0, 2, 3, 1))
        image = deprocess_img(image) * 255
        path_l = []
        for id in range(bs):
            cv2.imwrite(f"{id}.jpg", image[id])
            path_l.append(f"{id}.jpg")
    return path_l


def test_a_gan():
    G.load_state_dict(torch.load("G_net_param.pth"))
    demo = gr.Interface(
        fn=predict,
        inputs=["number"],
        outputs=gr.Gallery(label="result", columns=[4], rows=[4], height="auto")
    )
    demo.launch()

D = Discriminator().cuda()
G = Generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

#Training = True
Training = False

if Training:
    d_error, g_error = train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)
    fig, ax = plt.subplots()
    x = np.linspace(0, NUM_EPOCHS-1, NUM_EPOCHS)
    ax.plot(x, d_error, label='discriminator error')
    ax.plot(x, g_error, label='generator error')
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.legend()
    plt.savefig("training.png", dpi=300)
    print("fig saved.")
else:
    test_a_gan()