#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 01:49:13 2019

@author: subhashis
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import cv2
from torch.autograd import Variable
import numpy as np

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # replace the pre-trained head with a new one
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)
    return model

labels = {'person':0,'rider':1,'car':2,'truck':3,
         'bus':4,'motorcycle':5,'bicycle':6,'autorickshaw':7,'animal':8,'traffic light':9,
          'traffic sign':10,'vehicle fallback':11,'caravan':12,'trailer':13,'train':14}

model = get_model(len(labels))

ckpt=torch.load('model.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

image = cv2.imread('0001545.jpg')
image = np.swapaxes(image,0,2)
image = np.expand_dims(image, axis=0)

image = Variable(torch.from_numpy(image))
p = model(image)









