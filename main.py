import MyImg
import VAE
import Controler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

IMAGE_PATH="/home/cxr/homework/all/10.jpg"
VAE_INPUT_PATH="/home/cxr/homework/all/"
GRU_INPUT_PATH="/home/cxr/homework/separate/"
VAE_NET_PATH='vae/relu-bce-aug.pth'
GRU_NET_PATH='/gru/gru.pth'

#"no_data_augment-mse.pth"
#"no_data_augment-bce.pth"

#aug = transforms.ColorJitter(brightness=1)

def main():
    """ Controler.train_vae(
        load_net_path=VAE_NET_PATH,
        save_net_path=VAE_NET_PATH,
        data_path=VAE_INPUT_PATH
    ) """
    
    Controler.train_gru(
        vae_path=VAE_NET_PATH,
        load_gru_path=None,
        save_gru_path=GRU_NET_PATH,
        data_path=GRU_INPUT_PATH
    )

    """ img = MyImg.load_img(IMAGE_PATH)
    img = transforms.ToTensor()(img).unsqueeze(0)
    autoencoder.show_net(img) """

if __name__ == "__main__":
    main()