import VAE
import GRU
import MyImg

def train_vae(load_net_path=None,save_net_path=None,data_path=None):
    autoencoder = VAE.MyNet(load_net_path)
    imgs = MyImg.list_of_image(data_path) # tensor(11,3,419,179)
    autoencoder.train_net(imgs)
    if save_net_path != None:
        autoencoder.save_net(save_net_path)

def train_gru(vae_path=None,load_gru_path=None,save_gru_path=None,data_path=None):
    if vae_path == None:
        print("where is vae?")
        return
    vae = VAE.MyNet(vae_path)
    gru = GRU.MyNet(load_gru_path)
    batch_of_images = MyImg.batch_of_images(data_path)
    gru.train_net(batch_of_images,vae)
    if save_gru_path != None:
        gru.save_net(save_gru_path)