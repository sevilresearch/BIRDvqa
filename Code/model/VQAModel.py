import torch
import torchvision
from torch import mul, cat, tanh, relu
import os

class VQA_v1(torch.nn.Module):
  def __init__(self, embedding_size, num_answers):
    super(VQA_v1, self).__init__()

    #Image network
    #Outputs vector of shape 512
    resnet = torchvision.models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 1024),
        torch.nn.Tanh(),
        torch.nn.Linear(1024, 512),

    )

    self.mdl = resnet

    #question network 
    self.fc2 = torch.nn.Linear(embedding_size, 512)      
    self.fc3 = torch.nn.Linear(512, 512)                  

    # Layers for Merging
    self.fc4 = torch.nn.Linear(1024, 512)
    self.fc5 = torch.nn.Linear(512, num_answers)

  def forward(self, x, q):
    # The Image network
    x = self.mdl(x)                           

    # The question network
    act = torch.nn.Tanh()
    q = act(self.fc2(q))                      
    q = act(self.fc3(q))                       

    # Merge -> output
    out = cat((x, q), 1)                        # concat function
    out = act(self.fc4(out))                    # activation
    out = self.fc5(out)                         # output probability
    return out
