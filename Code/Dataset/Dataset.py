from DataHelper import read_images_path, read_questions
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import os
from torch.utils.data import Dataset
import glob
import numpy as np

IMAGE_PATH = "CarsDataset/train"

CLASS_DICT = {
   'SUV': 0,           
   'Cargo Truck': 1,
   'Sports Car': 2,
   'Pickup': 3
}

def preprocess_image(image_path):
    im = Image.open(image_path).convert('RGB')
    im = im.resize((512, 512))
    im = F.to_tensor(im)
    im = np.array(im)
    return im

def read_images(path_to_img):
    img = preprocess_image(path_to_img)
    return img

class CustomDataset(Dataset):
    def __init__(self, img_dir, all_images, txt, ans):
      self.img_dir = img_dir
      self.txt = txt
      self.ans = ans
      self.image_paths = glob.glob(f"{self.img_dir}/*.jpg")
      self.all_images = sorted(self.image_paths)
    def __len__(self):
        return len(self.ans)

    def __getitem__(self, idx):
      #Index divided by 7 since we are asking 7 questions per image
      img_dir = self.all_images[int(idx/7)]
      img = read_images(img_dir)
      ans = self.ans[idx]
      txt = self.txt[idx]
      return img, txt, ans


