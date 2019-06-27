import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import MyImg
import time
import random
from torch.autograd import Variable

EPOCH = 1
ITER = 1
RENDER = False
GPU = False
FEATURE = 48*25*10
ARG = [6,12,24,48]

class MyNet(nn.Module):
    def __init__(self, path=None):
        super(MyNet, self).__init__()

        # in_channels, out_channels, kernel_size, stride, padding
        # (3,419,179)->(6,209,89)->(12,104,44)->(24,51,21)
        """ self.en_conv = nn.Sequential(nn.Conv2d(3, ARG[0], 3, 2), nn.BatchNorm2d(ARG[0]),
                                     nn.Sigmoid(), nn.Conv2d(ARG[0], ARG[1], 3, 2),
                                     nn.BatchNorm2d(ARG[1]), nn.Sigmoid(),
                                     nn.Conv2d(ARG[1], ARG[2], 4, 2),
                                     nn.BatchNorm2d(ARG[2]), nn.Sigmoid()) """

        self.en_conv = nn.Sequential(nn.Conv2d(3, ARG[0], 3, 2), 
                                     nn.ReLU(),
                                     nn.Conv2d(ARG[0], ARG[1], 3, 2),
                                     nn.ReLU(),
                                     nn.Conv2d(ARG[1], ARG[2], 4, 2),
                                     nn.ReLU())

        # (24,51,21)->(48,25,10)
        self.en_conv_mean = nn.Sequential(nn.Conv2d(ARG[2], ARG[3], 3, 2),
                                          nn.BatchNorm2d(ARG[3]), nn.Tanh())
        self.en_conv_std = nn.Sequential(nn.Conv2d(ARG[2], ARG[3], 3, 2),
                                         nn.BatchNorm2d(ARG[3]), nn.Sigmoid())

        # (48,25,10)->(24,51,21)->(12,104,44)->(6,209,89)->(3,419,179)
        """ self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(ARG[3], ARG[2], 3, 2),
            nn.BatchNorm2d(ARG[2]),
            nn.Sigmoid(),
            nn.ConvTranspose2d(ARG[2], ARG[1], 4, 2),
            nn.BatchNorm2d(ARG[1]),
            nn.Sigmoid(),
            nn.ConvTranspose2d(ARG[1], ARG[0], 3, 2),
            nn.BatchNorm2d(ARG[0]),
            nn.Sigmoid(),
            nn.ConvTranspose2d(ARG[0], 3, 3, 2),
            nn.Sigmoid()
        ) """

        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(ARG[3], ARG[2], 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ARG[2], ARG[1], 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ARG[1], ARG[0], 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ARG[0], 3, 3, 2),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters())

        if path!=None:
            self.load_net(path)
        else:
            if GPU:
                self.cuda()

        print('vae has been initialized')

    def encoder(self, x):
        # x 四维图片数据 (batchsize, channel, hi, we)
        conv_mid = self.en_conv(x)
        conv_mean = self.en_conv_mean(conv_mid)
        conv_std = self.en_conv_std(conv_mid)

        # (batch,feature=12000)
        conv_mean = conv_mean.view(x.size(0), -1)
        conv_std = conv_std.view(x.size(0), -1)

        return conv_mean, conv_std

    def decoder(self, x):
        x = x.view(-1, ARG[3], 25, 10)
        out = self.de_conv(x)
        return out

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = Variable(eps)
        if GPU:
            eps = eps.cuda()
        code = eps.mul(var).add_(mean)
        return code

    def forward(self, x):
        mean, std = self.encoder(x)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        return out, code, mean, std

    def loss_f(self, out, target, mean, std):
        out_loss = F.binary_cross_entropy(out, target)
        #out_loss = F.mse_loss(out,target)
        if GPU:
            out_loss = out_loss.cuda()
        latent_loss = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())
        return out_loss #+ 0.00001*latent_loss

    def save_net(self,path):
        all_data = dict(
            optimizer = self.optimizer.state_dict(),
            model = self.state_dict(),
            info = u'vae参数'
        )
        torch.save(all_data,path)

    def load_net(self,path):
        all_data = torch.load(path)
        self.load_state_dict(all_data['model'])
        self.optimizer.load_state_dict(all_data['optimizer'])

    def show_net(self, input):
        output, _, _, _ = self(input)
        #i = random.randint(0,10)
        i = 0 
        img = output.data[i]
        MyImg.show_img(img)

    """ # data: Variable(11,3,419,179) """
    # data: a list of PIL.Image
    def train_net(self,imgs):
        input = None
        start = time.clock()

        for iter in range(ITER):
            # 每次迭代都对图像作新的变换
            if iter % 3 == 0:
                data = MyImg.augment_imgs(imgs)
                input = Variable(data)
                if GPU:
                    input = input.cuda()

            output, _, mean, std = self(input)
            loss = self.loss_f(output, input, mean, std)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter % 250 == 0:
                print('Iter: ', iter, ' | training loss: %.4f' % loss.item())
                if RENDER:
                    img = output.data[0]
                    MyImg.show_img(img)

        if not RENDER:
            elapsed = time.clock() - start
            print('training time: %.1f s' % elapsed)

        self.show_net(input)