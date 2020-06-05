# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:36:05 2020

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
from Architecture import *
import sys
import os
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

data_directory = sys.argv[1]
prediction = sys.argv[2]
target_dataset = ImageFolder(os.path.join(data_directory,'test_data'), transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
from torch.optim.lr_scheduler import StepLR

feature_extractor = Vgg16().cuda()
feature_extractor.load_state_dict(torch.load("extractor_model_vgg.bin"))
#summary(feature_extractor, (1, 32, 32))
label_predictor = LabelPredictor().cuda()
label_predictor.load_state_dict(torch.load('predictor_model_vgg.bin'))
#domain_classifier = DomainClassifier().cuda()

result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(prediction,index=False)
