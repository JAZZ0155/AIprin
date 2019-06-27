import torch
import torch.nn as nn
from torch.autograd import Variable
import VAE
import MyImg
import torch.nn.functional as F
import os

ITER = 10
GPU = False

class MyNet(nn.Module):
    def __init__(self,path):
        super(MyNet, self).__init__()

        self.gru = nn.GRU(
            input_size=VAE.FEATURE,
            hidden_size=VAE.FEATURE,
            batch_first=True
        )

        # h_0 (batch, num_layers * num_directions, hidden_size)
        self.h0 = Variable(torch.zeros(1,4,12000))

        self.optimizer = torch.optim.Adam(self.parameters())
        a=self.state_dict()
        if path!=None:
            self.load_net(path)
        else:
            if GPU:
                self.cuda()
        
        print('gru has been initialized')

    # features (batch=4,seq_len=4,input_size=12000)
    def forward(self, features):
        # h_n (batch=4, num_layers * num_directions=1, hidden_size=12000)
        # shape of h_n is the same as h_0
        _, h_n = self.gru(features,self.h0)
        return h_n

    def save_net(self,path):
        all_data = dict(
            optimizer = self.optimizer.state_dict(),
            model = self.state_dict(),
            info = u'gru参数'
        )
        torch.save(all_data,path)

    def load_net(self,path):
        all_data = torch.load(path)
        self.load_state_dict(all_data['model'])
        self.optimizer.load_state_dict(all_data['optimizer'])

    def loss_f(self, out, target):
        loss = F.mse_loss(out,target)
        return loss

    # features(batch=4,seq_len=4,input_size=12000), target(batch=4,input_size=12000)
    # inputs are Variables
    def train_net_once(self,features,target_feature):
        features = Variable(features)
        target_feature = Variable(target_feature)
        net_out = self(features).squeeze()

        loss = self.loss_f(net_out,target_feature)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    #boi:[[],[],[],[]]
    def train_net(self,batch_of_images,encoder):
        for iter in range(ITER):
            gru_input = torch.Tensor()
            gru_target = torch.Tensor()

            # 共4组images 得到 gru input和target
            for images in batch_of_images:
                vae_input = Variable(MyImg.augment_imgs(images)) # (5,3,419,179)
                _, features, _, _ = encoder(vae_input) # f : (5,12000)
                inp = features.data[0:4].unsqueeze(0) # (1,seq_len=4,input_size=12000)
                tar = features.data[4].unsqueeze(0) # (1,12000)
                gru_input = torch.cat([gru_input,inp],0)
                gru_target = torch.cat([gru_target,tar],0)

            loss = self.train_net_once(gru_input,gru_target)

            if iter % 250 == 0:
                print('Iter: ', iter, ' | training loss: %.4f' % loss.item())