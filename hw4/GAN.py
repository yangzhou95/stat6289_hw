import torch.nn as nn
import torch
import torch.optim
from torchvision.utils import save_image #save image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader #read data in batch
import torchvision.transforms as transforms # transform tensor
from torch.autograd import Variable
import time #timing
import torch.nn.init as init # init neural network




if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

#read dataset
def prepare_data():
    """"""
    dataset = CIFAR10(root = './data',
                     download = True, transform = transforms.ToTensor()) #download data
    #read data in batch. Total:50000, batch_size=64, n_batch=50000/64=781
    dataloader = DataLoader(dataset, batch_size= 64, shuffle= True)
    return dataset, dataloader

def build_discriminator():
    """"""
    n_d_feature = 64
    n_channel = 3
    dnet = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4,
                     stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4,
                     stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4,
                     stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4))
    print(dnet)
    return dnet

def build_generator(latent_size = 64, n_g_feature = 64, n_channel=3):
    """"""
    latent_size = 64
    n_g_feature = 64
    gnet = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4,
                              bias=False),
            nn.BatchNorm2d(4 * n_g_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4,
                              stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_g_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4,
                              stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_g_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4,
                              stride=2, padding=1),
            nn.Sigmoid())
    print(gnet)
    return gnet


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


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
    # start training
    start = time.time()  # start record time

    latent_size = 64
    n_g_feature = 64
    n_channel = 3
    dataset, dataloader = prepare_data()
    dnet = build_discriminator()
    gnet = build_generator(latent_size, n_g_feature)
    gnet.apply(weights_init)
    dnet.apply(weights_init)
    start_training(gnet, dnet, dataset, dataloader, latent_size, n_g_feature)
    end = time.time()
    print((end - start) / 60)  # output time for processing (unit: minute)
    # print("**************Finished************")
