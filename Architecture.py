# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:34:33 2020

@author: 王式珩
"""
import torch.nn as nn
class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

class Vgg16(nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        # input is torch.Size([64, 1, 32, 32])
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), #[64, 32, 32]
             nn.BatchNorm2d(64),
             nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, 1, 1),#[64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2),#[64, 16, 16]
            
            nn.Conv2d(64, 128, 3, 1, 1), #[128, 16, 16]
             nn.BatchNorm2d(128),
             nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),#[128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),#[128, 8, 8]
            
            nn.Conv2d(128, 256, 3, 1, 1),
             nn.BatchNorm2d(256),
             nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
             nn.BatchNorm2d(256),
             nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
  

            nn.Conv2d(256, 512, 3, 1, 1),
             nn.BatchNorm2d(512),
             nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
             nn.BatchNorm2d(512),
             nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
             nn.BatchNorm2d(512),
             nn.ReLU(),
            nn.MaxPool2d(2),
           
            nn.Conv2d(512, 512, 3, 1, 1),
             nn.BatchNorm2d(512),
             nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
             nn.BatchNorm2d(512),
             nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        
        x = self.conv(x).squeeze()
        
        # x = torch.flatten(x)
        return x