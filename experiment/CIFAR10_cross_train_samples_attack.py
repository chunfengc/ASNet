#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import time
import pdb
import os

import sys
sys.path.append('../')
from model.vgg import *
from ASNet.utils import *
from ASNet.AS_Attack import *
from ASNet.ASModel import *
from ASNet.PCEModel import *
from ASNet.FineTuning import *
from ASNet.ASNet import *
from datasets.dataloader_cifar import *
from ASNet.UAP import *
import torch.nn.functional as F


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device = torch.device('cpu')
# model
model = vgg19_bn(num_classes=10)
pretrained = '../model/CIFAR10/vgg19_bn/model_best.pth.tar'
if torch.cuda.is_available():
    tmp = torch.load(pretrained)
else:
    tmp = torch.load(pretrained, torch.device('cpu'))
model.load_state_dict({''.join(k.split('.module')) : v for k,v in tmp['state_dict'].items()})
model.to(device).eval()

 
#dataset
class_idx = 1
bs = 100
train_loader_one_class = trainloader_cifar_vgg19(bs,shuffle=False,class_label=class_idx)
test_loader_one_class = testloader_cifar_vgg19(bs,class_label=class_idx)
train_loader = trainloader_cifar_vgg19(bs,shuffle=False)
test_loader = testloader_cifar_vgg19(bs)
 

image_shape = [1,*train_loader_one_class.dataset[0][0].shape]
# Proposed AS 

ntrain_attacks = [10,30,50,100,200]
nrepeat = 10

num_attack = len(ntrain_attacks) 
Training_fool_ratio = torch.zeros([nrepeat, num_attack,2 ])
Testing_fool_ratio = torch.zeros([nrepeat, num_attack,2 ])
num_batches = [len(test_loader_one_class),len(test_loader)]
num_max_sampels = [len(train_loader_one_class.dataset),len(train_loader.dataset)]

noise_level = 10

for ip in range(nrepeat): 
    for idx_n,ntrain_attack in enumerate(ntrain_attacks): 
        for idx_c in range(2): 
            randidx = np.random.randint(num_max_sampels[idx_c], size=(ntrain_attack))
            dataset = torch.zeros([ntrain_attack,*image_shape[1:]])
            dataset_labels = torch.tensor([class_idx]).repeat(ntrain_attack)
 
            if idx_c ==0:
                for i in range(dataset.shape[0]):
                    dataset[i] = train_loader_one_class.dataset[randidx[i]][0]
                    dataset_labels[i] = train_loader_one_class.dataset[randidx[i]][1]
            else:
                for i in range(dataset.shape[0]):
                    dataset[i] = train_loader.dataset[randidx[i]][0]
                    dataset_labels[i] = train_loader.dataset[randidx[i]][1]                
            
            v_AS_attack = AS_attack_input_small_dataset_sz(model,dataset,dataset_labels,
                                     noise_level = noise_level, loss=F.cross_entropy,
                                     attack_target = None,device=device, szmax=3) 

            model,dataset,dataset_labels = model.to(device), dataset.to(device),dataset_labels.to(device)
            Training_fool_ratio[ip, idx_n,idx_c] = torch.sum(model(
                noise_level*v_AS_attack+dataset).argmax(dim=1)!=dataset_labels) 

            if idx_c ==0:
                for idx, (image_data, image_labels) in enumerate(test_loader_one_class): # Testing
                    image_data, image_labels = image_data.to(device), image_labels.to(device)
                    Testing_fool_ratio[ip,idx_n,idx_c] += torch.sum(model(
                        noise_level*v_AS_attack+image_data).argmax(dim=1)!=image_labels) 
                    if idx+1 >= num_batches[idx_c]:
                        break
            else:
                for idx, (image_data, image_labels) in enumerate(test_loader): # Testing
                    image_data, image_labels = image_data.to(device), image_labels.to(device)
                    Testing_fool_ratio[ip,idx_n,idx_c] += torch.sum(model(
                        noise_level*v_AS_attack+image_data).argmax(dim=1)!=image_labels) 
                    if idx+1 >= num_batches[idx_c]:
                        break
            
            Training_fool_ratio[ip,idx_n,idx_c] = (
                Training_fool_ratio[ip,idx_n,idx_c]/ntrain_attack) 
            Testing_fool_ratio[ip,idx_n,idx_c]  = (
                Testing_fool_ratio[ip,idx_n,idx_c]/num_batches[idx_c]/bs)
            
            print(ntrain_attack, Training_fool_ratio[ip,idx_n,idx_c], Testing_fool_ratio[ip,idx_n,idx_c])
         
        filename = './results/CIFAR10_cross_train_number_samples_attack.pth'
        torch.save([ntrain_attacks, Training_fool_ratio, Testing_fool_ratio,class_idx],filename)
