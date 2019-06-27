import cv2
import numpy as np
import os
import torch
import VAE
from torchvision import transforms
from PIL import Image

BATCH = 11

# from PIL to Tensor
""" augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
]) """
augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.ToTensor()
])

to_tensor = transforms.ToTensor()

# return a batch of processed images (Tensor)
def pre_process(path):
    out = torch.Tensor()
    for _,_,files in sorted(os.walk(path)):
        for file in sorted(files):
            img = load_img(os.path.join(path,file)) #PIL.Image
            img = augment(img).unsqueeze(0)
            #img = to_tensor(img).unsqueeze(0)
            out = torch.cat([out,img],0)
    return out

# return list of PIL.Image
def list_of_image(path):
    out = []
    for _,_,files in sorted(os.walk(path)):
        for file in sorted(files):
            img = load_img(os.path.join(path,file))
            out.append(img)
    return out

def batch_of_images(path):
    batch_of_images = []
    # 读路径下所有文件夹
    dirs = [d.name for d in os.scandir(path) if d.is_dir()]
    # 存储所有原图像 [[],[],[],[]]
    for dir in sorted(dirs):
            images = list_of_image(os.path.join(path,dir))
            batch_of_images.append(images)
    return batch_of_images

# list(Image)->Tensor
def augment_imgs(imgs):
    out = torch.Tensor()
    for img in imgs:
        img = augment(img).unsqueeze(0)
        out = torch.cat([out,img],0)
    return out

# return PIL.Image
def load_img(file):
    # (4160,2768)->(419,179) 横向裁剪并缩小
    img = Image.open(file)
    img = img.crop((554,0,2330,4158))
    img.thumbnail((179,419))
    """ img = cv2.imread(file) # type(img) = ndarray (height, weight, channel)
    img = cv2.resize(img, (290, 419))
    img = img[:,61:240,:] # (419,179,3) """
    return img

# input a tensor
def show_img(img):
    # method in plt (rgb)
    """ plt.imshow(img)
    plt.show() """
    # method in cv (bgr)
    """ cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  """
    # method in PIL
    img = transforms.ToPILImage()(img)
    img.show()

# (i,j,k)->(k,i,j)
def cv_to_torch(img):
    arr = np.transpose(img,(2,0,1))
    """ arr = np.empty(shape=(img.shape[2],img.shape[0],img.shape[1]),dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                arr[k][i][j]=img[i][j][k] """
    return arr

# (i,j,k)->(j,k,i)
def torch_to_cv(arr):
    img = np.transpose(arr,(1,2,0))
    """ img = np.empty(shape=(arr.shape[1],arr.shape[2],arr.shape[0]),dtype=np.uint8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                img[j][k][i]=arr[i][j][k] """
    return img
