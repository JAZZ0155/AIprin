import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

INPUT_PATH="/home/cxr/homework/all"

train_set = datasets.ImageFolder(root=INPUT_PATH, transform=transforms.ToTensor)
dataloader = DataLoader(train_set, batch_size = 1)
to_pil_image = transforms.ToPILImage()

a = iter(dataloader)
img = a.next()
#img = to_pil_image(img[0])
img.show()

""" for step, (data,target) in enumerate(dataloader):
    img = to_pil_image(data[0])
    img.show() """