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
num_classes=100
model = vgg19_bn(num_classes=num_classes)
pretrained = '../model/CIFAR100/vgg19_bn/model_best.pth.tar'
if torch.cuda.is_available():
    tmp = torch.load(pretrained)
else:
    tmp = torch.load(pretrained, torch.device('cpu'))
model.load_state_dict({''.join(k.split('.module')) : v for k,v in tmp['state_dict'].items()})
#model.load_state_dict(tmp['state_dict'])
model.to(device).eval()

 
#dataset
class_idx = 1
bs = 100
train_loader_one_class = trainloader_cifar_vgg19(bs, shuffle=False,
                                                 class_label=class_idx, num_class=num_classes)
test_loader_one_class = testloader_cifar_vgg19(bs, class_label=class_idx, num_class=num_classes)
 
compute_loss(model, device, test_loader_one_class,is_print=True)
image_shape = [1,*train_loader_one_class.dataset[0][0].shape]
# Proposed AS 

ntrain_attack = 100
nrepeat = 10
num_noise_level = 6
noise_levels = np.linspace(5,10,num=num_noise_level)
       
Training_fool_ratio = torch.zeros([nrepeat,3,num_noise_level])
Testing_fool_ratio = torch.zeros([nrepeat,3,num_noise_level])
num_batch = len(test_loader_one_class)
num_max_sampels = len(train_loader_one_class.dataset)
dataset_labels = torch.tensor([class_idx]).to(device)

v_ASs = torch.zeros([nrepeat, *image_shape[1:]])
v_UAPs = torch.zeros([nrepeat, *image_shape[1:]])
v_rands = torch.zeros([nrepeat, *image_shape[1:]])
cpu_times = torch.zeros([nrepeat,2,num_noise_level])

for ip in range(nrepeat):
    
    randidx = np.random.randint(num_max_sampels, size=(ntrain_attack))
    dataset = torch.zeros([ntrain_attack,*image_shape[1:]])
    for i in range(ntrain_attack):
        dataset[i] = train_loader_one_class.dataset[randidx[i]][0]
    
    for idx_n,noise_level in enumerate(noise_levels): 
        v_AS_starts = time.time()
        v_AS_attack = AS_attack_input_small_dataset_sz(model,dataset,dataset_labels,device=device,
                                noise_level = noise_level, loss=F.cross_entropy,attack_target = None)
        v_ASs[ip] = v_AS_attack

        v_AS_ends = time.time()
        cpu_times[ip,0,idx_n] = v_AS_ends-v_AS_starts
        print('   AS attack vector computed in {:.2f} s'.format(cpu_times[ip,0,idx_n]))
        # UAP [CVPR 2017]

        v_UAP = universal_perturbation(dataset, model,dataset_labels,xi=noise_level, device=device)
        v_UAP /= torch.norm(v_UAP)
        v_UAPs[ip] = v_UAP
        cpu_times[ip,1,idx_n] = time.time()-v_AS_ends
        print('   UAP attack vector computed in {:.2f} s'.format(cpu_times[ip,1,idx_n]))


        # random
        v_rand = torch.randn(image_shape).to(device)
        v_rand /= torch.norm(v_rand)
        v_rands[ip] = v_rand

        dataset = dataset.to(device)
        model = model.to(device)
        
        #for image_data in dataset: # Training
        Training_fool_ratio[ip,0,idx_n] = torch.sum(model(
            noise_level*v_AS_attack+dataset).argmax(dim=1)!=dataset_labels)
        Training_fool_ratio[ip,1,idx_n] = torch.sum(model(
            noise_level*v_UAP+dataset).argmax(dim=1)!=dataset_labels)
        Training_fool_ratio[ip,2,idx_n] = torch.sum(model(
            noise_level*v_rand+dataset).argmax(dim=1)!=dataset_labels)
        
        for idx, (image_datas, image_labels) in enumerate(test_loader_one_class): # Testing
            image_datas, image_labels = image_datas.to(device), image_labels.to(device)
            Testing_fool_ratio[ip,0,idx_n] += torch.sum(model(
                noise_level*v_AS_attack+image_datas).argmax(dim=1)!=image_labels)
            Testing_fool_ratio[ip,1,idx_n] += torch.sum(model(
                noise_level*v_UAP+image_datas).argmax(dim=1)!=image_labels)
            Testing_fool_ratio[ip,2,idx_n] += torch.sum(model(
                noise_level*v_rand+image_datas).argmax(dim=1)!=image_labels)

            if idx+1 >= num_batch:
                break
        print(noise_level, Training_fool_ratio[ip,:,idx_n], Testing_fool_ratio[ip,:,idx_n])

        Training_fool_ratio[ip,:,idx_n] /= ntrain_attack
        Testing_fool_ratio[ip,:,idx_n] /=(num_batch*bs)

        filename = './results/CIFAR100_VGG19_UAP_AS_random_Attack_one_class_{}.pth'.format(class_idx)
        torch.save([noise_levels, Training_fool_ratio, Testing_fool_ratio,v_ASs,v_UAPs,
                    v_rands,cpu_times],filename)
