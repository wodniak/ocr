from models.resnet import *
import torch
from torchsummary import summary

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

model = resnet18(pretrained=True, progress=True)

summary(model)
