from urllib.request import urlopen
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from os import path
import sys


from VQAModel import VQA_v1
from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer('all-mpnet-base-v2')

def get_answers(fileName):
  with open(fileName, 'r') as answers_file:

    return [a.strip() for a in answers_file]

def load_and_process_image_url(url):
    # Loads image from path and converts to Tensor, you can also reshape the im
    im = Image.open(urlopen(url))
    im = F.to_tensor(im)
    return im



model = VQA_v2(embedding_size = 768, num_answers = 431)
model.load_state_dict(torch.load("../model.pt")) 
model.eval()

all_answers = get_answers('../answers.txt')

path = "../CarsDataset/car_data/car_data/test/00840.jpg"
image = F.to_tensor(Image.open(path).resize((512,512)))
image = image.unsqueeze(0)

text = 'what car is this?'
text = st_model.encode(text)
text = torch.tensor(text, dtype=torch.float)
text = text.unsqueeze(0)

probs = model.forward(image, text)
answer_idx = torch.argmax(probs, dim=1) # index of answer with highest probability
answer_text = [all_answers[idx] for idx in answer_idx] # convert index to answer text
print(answer_text)