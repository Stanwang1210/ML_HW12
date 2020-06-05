# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:25:04 2020

@author: 王式珩
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import sys
import os
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
source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
#torch.save(source_dataloader, 'source_dataloader.pth')
#torch.save(target_dataloader, 'target_dataloader.pth')

feature_extractor = Vgg16().cuda()
feature_extractor.load_state_dict(torch.load("extractor_model_vgg.bin"))
#summary(feature_extractor, (1, 32, 32))
label_predictor = LabelPredictor().cuda()
label_predictor.load_state_dict(torch.load('predictor_model_vgg.bin'))
domain_classifier = DomainClassifier().cuda()

feature_extract = []
answer = []
label_predictor.eval()
feature_extractor.eval()
for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        domain_logits = domain_classifier(feature.detach())
        feature_extract.append(domain_logits.detach().numpy())
        answer.append(domain_label.detach().numpy())
        # class_logits = label_predictor(feature[:source_data.shape[0]])
        # domain_logits = domain_classifier(feature)
        # loss = domain_criterion(domain_logits, domain_label)
        # running_D_loss+= loss.item()
        # loss.backward()
        # optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        # loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        # running_F_loss+= loss.item()
        # loss.backward()
        # optimizer_F.step()
        # optimizer_C.step()

        # optimizer_D.zero_grad()
        # optimizer_F.zero_grad()
        # optimizer_C.zero_grad()

        # total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        # total_num += source_data.shape[0]

        
        # print(i, end='\r')

    # return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num
# for i, (test_data, _) in enumerate(test_dataloader):
#     test_data = test_data.cuda()

#     class_logits = label_predictor(feature_extractor(test_data))

#     x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
#     result.append(x)

# import pandas as pd
# result = np.concatenate(result)

# # Generate your submission
# df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
# df.to_csv(prediction,index=False)
import numpy as np
np.save('feature_extract.npy',np.array(feature_extract))
np.save('answer.npy',np.array(answer))
