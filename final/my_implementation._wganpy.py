import time #timing
# import argparse
import os
import numpy as np
import math
import sys

from torchvision.utils import save_image #save image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader #read data in batch
import torchvision.transforms as transforms #
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init # init neural network

# recursively create dir. With exist_ok=False, if dir exist, it
# triggers OSError; with exist_ok=True, no error.
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.model = nn.Sequential(
        #     *block(opt.latent_dim, 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(img_shape))),
        #     nn.Tanh())
        self.model = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=64, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)
        self.output = nn.Tanh()
    def forward(self, z):
        img = self.model(z)
        img =self.output(img)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),#np.prod(img_shape)height*width*depth
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # improvement 1. remove the sigmoid in the last layer as
            # sigmoid may result in vanishing gradients
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

def prepare_data():
    """"""
    dataset = CIFAR10(root = './datasets/cifar/',
                     download = True, transform = transforms.ToTensor()) #download data
    #read data in batch. Total:50000, batch_size=64, n_batch=50000/64=781
    dataloader = DataLoader(dataset, batch_size= 64, shuffle= True)
    return dataset, dataloader

def start_training(gnet, dnet,dataset, dataloader, latent_size = 64, n_g_feature = 64):
    """"""
    #loss
    criterion = nn.BCEWithLogitsLoss()
    #optimizer
    goptimizer = torch.optim.Adam(gnet.parameters(),
                                 lr=0.0002, betas=(0.5, 0.999))
    doptimizer = torch.optim.Adam(dnet.parameters(),
                                 lr=0.0002, betas=(0.5, 0.999))

    #gnerate noise and feed it into the Generator network
    batch_size = 64
    fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    epoch_num = 20  # training epochs
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(dataloader):
            real_images, _ = data
            batch_size = real_images.shape[0]
            # train the discriminator D
            labels = torch.ones(batch_size)  # true data label: 1
            preds = dnet(Variable(real_images.type(Tensor)))  # feed the true data to discriminator D
            outputs = preds.reshape(-1)  # reshape
            dloss_real = criterion(outputs, labels.type(Tensor))
            dmean_real = outputs.sigmoid().mean()  # output the percentage of Discriminator that label true data as True

            noises = torch.randn(batch_size, latent_size, 1, 1)
            fake_images = gnet(noises.type(Tensor))  # generate fake image
            labels = torch.zeros(batch_size)  # generate label of fake image: 0
            fake = fake_images.detach()  # fix the params of generator
            preds = dnet(fake)  # feed the fake data to the discriminator
            outputs = preds.reshape(-1)  # reshape
            dloss_fake = criterion(outputs.type(Tensor), labels.type(Tensor))
            dmean_fake = outputs.sigmoid().mean()  # output the percentage of Discriminator that label fake data as fake

            dloss = dloss_real + dloss_fake  # the loss of discriminator: summation over two cases
            dnet.zero_grad()  # set gradient to zero
            dloss.backward()  # backpropagation
            doptimizer.step()

            # train the generator
            labels = torch.ones(batch_size)  # train generator G
            preds = dnet(fake_images)  # feed fake image to discriminator
            outputs = preds.reshape(-1)  # reshape
            gloss = criterion(outputs.type(Tensor), labels.type(Tensor))
            gmean_fake = outputs.sigmoid().mean()  # percentage of labeling fake data as true

            gnet.zero_grad()  # set gradient to zero
            gloss.backward()  # backpropagation
            goptimizer.step()

            # output the training performance
            print('[{}/{}]'.format(epoch, epoch_num) + '[{}/{}]'.format(batch_idx, len(dataloader)) +
                  'Loss of D:{:g} Loss of G：{:g}'.format(dloss, gloss) +
                  '% True-->True:{:g}, % Fake-->True:{:g}/{:g}'.format(dmean_real, dmean_fake, gmean_fake))
            if batch_idx % 100 == 0:
                fake = gnet(fixed_noise)  # use noise to generator fake image
                path = './data_new/gen{:02d}_batch{:03d}.png'.format(epoch, batch_idx)
                save_image(fake, path, normalize=False)

    # torch.save(dnet.state_dict(),'./D.pth')
    # torch.save(gnet.state_dict(),'./G.pth')

if __name__ == "__main__":
    start = time.time()
    dataset, dataloader = prepare_data()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    start_training(generator, discriminator, dataset, dataloader,latent_size=opt.latent_dim,n_g_feature=64)
    end = time.time()
    print((start-end)/60

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# Modification: chaneg the LSUN bedroom dataset to cifar
os.makedirs("./datasets/cifar/", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
data_set = datasets.CIFAR10("./datasets/cifar/", train=True, download=True,transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=data_set, batch_size=opt.batch_size, shuffle=True)

# Optimizers
# improvement 4. do not use momentum-based optimizer, RMSProp and SGD are preferred
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



# ********************Training****************
batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):
        #imags:64,3,32,32
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ******************** Train Discriminator ****************
        optimizer_D.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        z = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)

        # Generate a batch of images
        # detach means that the image is removed from the computation graph, and
        # it won't compute the gradient of generator anymore.-->save memory
        fake_imgs = generator(z).detach()
        # Adversarial loss
        # improvement 2. do not use the log loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        loss_D.backward()
        # only update the parameter of discriminator
        optimizer_D.step()

        # improvement 3.Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # ******************* Train Generator *******************
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()# only update parameters of generator

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1
        if i % 100 == 0:
            fake = generator(z)  # use noise to generator fake image
            path = './image_wgan/gen{:02d}_batch{:03d}.png'.format(epoch, i)
            save_image(fake, path, normalize=False)
