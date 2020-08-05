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
from ASNet.AS_Attack import *
from model.resnet import *
from ASNet.utils import *
from ASNet.ASModel import *
from ASNet.PCEModel import *
from ASNet.FineTuning import *
from ASNet.ASNet import *
from datasets.dataloader_cifar import *
from ASNet.UAP import *
import torch.nn.functional as F
from collections import OrderedDict 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device = torch.device('cpu')
 
class_idx = 1
bs = 100

 
modelnames = ['resnet20','resnet44','resnet56','resnet110','vgg19']
num_model  = len(modelnames)

ntrain_attack = 50
nrepeat = 10
 
Training_fool_ratio = torch.zeros([nrepeat,num_model])
Testing_fool_ratio = torch.zeros([nrepeat,num_model])
num_max_sampels = 5000
num_batch = min(10, np.floor(1000/bs))
   
dataset_labels = torch.tensor([class_idx])

v_ASs = torch.zeros([nrepeat, num_model,3,32,32])
noise_level = 10

is_training = True
is_validate_cross_model = True

def validate_error(test_loader_one_class,model,v_AS_attack,class_idx,noise_level=10,
                   device='cpu',num_batch=10):
    Testing_ratio = 0.0
    num_samples = 0
    model = model.to(device) 
    for idx, (image_data, image_label) in enumerate(test_loader_one_class): # Testing
        Testing_ratio += torch.sum(model(
            noise_level*v_AS_attack.to(device)+image_data.to(device)).argmax(dim=1)!=class_idx)
        num_samples += len(image_label)
        if idx+1>=num_batch:
            break
                  
    return Testing_ratio

def load_model(modelname):
    if modelname == 'vgg19':
        model = vgg19_bn(num_classes=10)
        pretrained = '../model/CIFAR10/vgg19_bn/model_best.pth.tar'
        if torch.cuda.is_available():
            tmp = torch.load(pretrained)
        else:
            tmp = torch.load(pretrained, torch.device('cpu'))
        model.load_state_dict({''.join(k.split('.module')) : v for k,v in tmp['state_dict'].items()})
        model.to(device).eval() 
        train_loader_one_class = trainloader_cifar_vgg19(bs,shuffle=False,class_label=class_idx)
        test_loader_one_class = testloader_cifar_vgg19(bs,class_label=class_idx)

    else:
        
        if modelname == 'resnet110':
            model = resnet110()
        elif modelname == 'resnet56':    
            model = resnet56()
        elif modelname == 'resnet44':
            model = resnet44()
        elif modelname == 'resnet20':
            model = resnet20()

        pretrained = '../model/ResNet/{}.th'.format(modelname)
        # tmp = torch.load(pretrained)
        # model.load_state_dict(tmp['state_dict'])
        if torch.cuda.is_available():
            tmp = torch.load(pretrained)
        else:
            tmp = torch.load(pretrained, torch.device('cpu'))
        sd = OrderedDict()
        for key, item in tmp['state_dict'].items():
            key_s = key.split('.', 1)[1]
            sd[key_s] = item 
        model.load_state_dict(sd)
        model = model.to(device)
        model.eval() 
        train_loader_one_class = trainloader_resnet(bs,class_label=class_idx)
        test_loader_one_class = testloader_resnet(bs,class_label=class_idx)

    
    return model,train_loader_one_class,test_loader_one_class

if is_training:
    for idx_n, modelname in enumerate(modelnames): 
        
        model,train_loader_one_class,test_loader_one_class = load_model(modelname)

        compute_loss(model, device, test_loader_one_class,is_print=True)
        image_shape = [1,*train_loader_one_class.dataset[0][0].shape]


        for ip in range(nrepeat):

            randidx = np.random.randint(num_max_sampels, size=(ntrain_attack))
            dataset = torch.zeros([ntrain_attack,*image_shape[1:]])
            for i in range(dataset.shape[0]):
                dataset[i] = train_loader_one_class.dataset[randidx[i]][0]


            v_AS_attack = AS_attack_input_small_dataset_sz(model,dataset,dataset_labels,
                                    noise_level = 10, loss=F.cross_entropy,attack_target = None,
                                    szmax = 3, )
            v_ASs[ip,idx_n] = v_AS_attack


            Training_fool_ratio[ip,idx_n] = torch.sum(model(
                noise_level*v_AS_attack+dataset).argmax(dim=1)!=class_idx)

             
            Testing_fool_ratio[ip,idx_n] += validate_error(test_loader_one_class,
                                       model,v_AS_attack,class_idx,noise_level=10,num_batch=num_batch)
              

            print(modelname, Training_fool_ratio[ip,idx_n], Testing_fool_ratio[ip,idx_n])

    Training_fool_ratio = (Training_fool_ratio/ntrain_attack).numpy()
    Testing_fool_ratio = (Testing_fool_ratio/num_batch/bs).numpy()


    filename = './results/CIFAR10_cross_model_VGG19_UAP_AS_random_Attack_one_class_{}.pth'.format(
        class_idx)
    torch.save([modelnames, Training_fool_ratio, Testing_fool_ratio,v_ASs],filename)
else:
    filename = './results/CIFAR10_cross_model_VGG19_UAP_AS_random_Attack_one_class_{}.pth'.format(
        class_idx)
    [modelnames, Training_fool_ratio, Testing_fool_ratio,v_ASs] = torch.load(filename)

                    
### computing the cross-model accuracy
if is_validate_cross_model:
    cross_attack_ratio = torch.zeros([num_model,num_model]) 
    for idx_n, modelname in enumerate(modelnames): 
        model,_,test_loader_one_class = load_model(modelname) 
         
        for im in range(num_model):
            for ip in range(nrepeat):
                v_AS_attack = v_ASs[ip,im] 
                cross_attack_ratio[idx_n,im] += validate_error(test_loader_one_class,model,
                                                   v_AS_attack,class_idx,noise_level=10,device=device)

        print(cross_attack_ratio[idx_n,:])
    cross_attack_ratio = cross_attack_ratio.double()/nrepeat/1000

filename = './results/CIFAR10_cross_model_error_Attack_one_class_{}.pth'.format(
        class_idx)
torch.save([modelnames, cross_attack_ratio ,v_ASs],filename)
