import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image as pil_image
import glob
from torchvision import transforms

def make_dataset(image_list):
    images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def get_transform(img_size, normalize):
    
    return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

def loader(video, img_size, normalize):
    images = []
    for path in glob.glob(f"{video}/*.png"):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess = get_transform(img_size, normalize)
        preprocessed_image = preprocess(pil_image.fromarray(image))
        images.append(preprocessed_image)
    return torch.stack(images)

        
class ImageList(Dataset):
    def __init__(self, image_list, normalize, img_size=299):
        imgs = make_dataset(image_list)
        self.imgs = imgs
        self.img_size = img_size
        self.normalize = normalize
    
    def __getitem__(self, index):
        direc, target = self.imgs[index]
        imgs = loader(direc, self.img_size, self.normalize)
        
        return imgs, target
    
    def __len__(self):
        return len(self.imgs)