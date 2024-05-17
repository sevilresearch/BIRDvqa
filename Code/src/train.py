import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import os
from os import path
import pandas as pd
from random import randint
import json
import torchvision.utils as utils
import numpy as np
import sys


from VQAModel import VQA_v1
from Dataset import CustomDataset

from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda")

if torch.cuda.is_available(): 
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(device))




def read_questions(question_path):
    with open(question_path, 'r') as file:
        qs = json.load(file)

    questions = [q[0] for q in qs]
    answers = [q[1] for q in qs]
    image_ids = [q[2] for q in qs]
    return questions, answers, image_ids

def get_answers(fileName):
  with open(fileName, 'r') as answers_file:

    return [a.strip() for a in answers_file]
  

train_questions, train_answers, train_image_ids = read_questions("../questions_and_answers.txt")
test_questions, test_answers, test_image_ids = read_questions("../questions_and_answers_test.txt")
all_answers = get_answers('../answers.txt')
num_answers = len(all_answers)


st_model = SentenceTransformer('all-mpnet-base-v2')

#Questions are encoded by calling model.encode()
train_X_seqs = st_model.encode(train_questions)
test_X_seqs = st_model.encode(test_questions)

# convert ndarray to tensor
train_X_seqs = torch.tensor(train_X_seqs, dtype=torch.float)
test_X_seqs = torch.tensor(test_X_seqs, dtype=torch.float)

print(f'\nThe shape of the binary vectors is : {train_X_seqs.shape}')

print('\n--- Creating model outputs...')
print()

train_answer_indices = np.array([all_answers.index(a) for a in train_answers])
test_answer_indices = np.array([all_answers.index(a) for a in test_answers])

#creating a 2D array filled with 0's
train_Y = np.zeros((train_answer_indices.size, train_answer_indices.max()+1), dtype=int)
test_Y = np.zeros((test_answer_indices.size, test_answer_indices.max()+1), dtype=int)

#replacing 0 with a 1 at the index of the original array
train_Y[np.arange(train_answer_indices.size),train_answer_indices] = 1
test_Y[np.arange(test_answer_indices.size),test_answer_indices] = 1

# finally convert the label vectors to tensor and fix the data type so it wouldnt error in the fully connected layer
train_Y = torch.tensor(train_Y, dtype=torch.float)
test_Y = torch.tensor(test_Y, dtype=torch.float)

print(f'Example model output: {train_Y[0]}')
print(f'data type {type(train_Y)}')

def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    model.to(device)
    total_loss, total = 0, 0
    count = 0

    for image, text, label in trainloader:
        count += 1
        
        
        # get the inputs; data is a list of [inputs, labels]
        image, text, label =  image.to(device), text.to(device), label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model.forward(image, text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Record metrics
        total_loss += loss.item()
        total += len(label)

    return total_loss / total


def validate_loop(model, criterion, valid_loader):
    model.eval()
    model.to(device)
    total_loss, total = 0, 0

    with torch.no_grad():
      for image, text, label in testloader:
          
          # get the inputs; data is a list of [inputs, labels]
          image, text, label =  image.to(device), text.to(device), label.to(device)

          # Forward pass
          output = model.forward(image, text)

          # Calculate how wrong the model is
          loss = criterion(output, label)

          # Record metrics
          total_loss += loss.item()
          total += len(label)

    return total_loss / total

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



if torch.cuda.is_available(): device = torch.device("cuda:0")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


# training and test datasets and data loaders

train_dataset = CustomDataset("../CarsDataset/car_data/car_data/train/", train_image_ids,train_X_seqs, train_Y)
test_dataset = CustomDataset("../CarsDataset/car_data/car_data/test/",test_image_ids, test_X_seqs, test_Y)
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
testloader = DataLoader(test_dataset, batch_size=4)


# Initialize model, recursively go over all modules and convert their parameters and buffers to CUDA tensors
model = VQA_v1(embedding_size = 768, num_answers = num_answers).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.5 )


train_losses, valid_losses = [], []

for epoch in range(5):
    train_loss = train_loop(model, optimizer, criterion, trainloader)
    valid_loss = validate_loop(model, criterion, testloader)

    tqdm.write(
        f'epoch #{epoch + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',
    )

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'Epoch: {epoch},\n')
    print(f'Training Loss: {train_loss},\n')
    print(f'Validation Loss: {valid_loss}\n')


import matplotlib.pyplot as plt
plt.style.use('ggplot')


epoch_ticks = range(1, epoch + 2)
plt.plot(epoch_ticks, train_losses)
plt.plot(epoch_ticks, valid_losses)
plt.legend(['Train Loss', 'Valid Loss'])
plt.title('Losses')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.xticks(epoch_ticks)
plt.show()

model.eval()
model.to(device)
num_correct = 0
num_samples = 0
predictions = []
answers = []

with torch.no_grad():
    for image, text, label in testloader:
        image, text, label =  image.to(device), text.to(device), label.to(device)
        probs = model.forward(image, text)

        _, prediction = probs.max(1)
        predictions.append(prediction)

        answer = torch.argmax(label, dim=1)
        answers.append(answer)

        num_correct += (prediction == answer).sum()
        num_samples += prediction.size(0)

    valid_acc = (f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    print(valid_acc)


torch.save(model.state_dict(), 'model2.pt')

