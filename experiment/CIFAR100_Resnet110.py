import os
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
 
import time
import sys
sys.path.append('../') 
from datasets.dataloader_cifar import *
from ASNet.utils import *
from ASNet.PCEModel import *
from ASNet.ASModel import *
from ASNet.FineTuning import *
from ASNet.ASNet import *
from model.resnet import *# resnet20,resnet32,resnet44,resnet56,resnet110#1202
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#torch.cuda.set_device(0)
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device = torch.device('cpu')

bs = 128
num_class = 100

train_loader = trainloader_resnet(bs,num_class)
test_loader = testloader_resnet(bs,num_class)
# test_labels = torch.tensor(test_loader.dataset.targets)
train_labels = torch.tensor(train_loader.dataset.targets)

modelname = 'resnet110'
model = resnet110(num_class)
    
pretrained = '../model/ResNet/model_resnet110_cifar100.th'
if torch.cuda.is_available():
    tmp = torch.load(pretrained)
else:
    tmp = torch.load(pretrained, torch.device('cpu'))
model.load_state_dict(tmp['state_dict'])
model.to(device)
model.eval()
seq_model = get_seq_model_resnet_nomodule(model)
seq_model.to(device)

train_max_batch = len(train_loader)

nAS = 50
r_max = max(200,nAS)
loss = F.cross_entropy
add_softmax = False
p = 2

init_cut = 35
last_cut = 45
fAS_name = './results/CIFAR100_resnet110_ASModel_nAS_%d_cut_%d_%d.pth'%(nAS,init_cut,last_cut)
if os.path.isfile(fAS_name):
    AS_models, Sigma = torch.load(fAS_name,map_location=device)
    print('    AS layer loaded')
else:
    AS_models, Sigma = get_ASModel_FD(seq_model, train_loader, 
           list(range(init_cut, last_cut)), train_max_batch,r_max,device)
    torch.save([AS_models,Sigma], fAS_name)
    print('    AS layer computed')
    
    
for cut_idx in range(41,45,3):
    cut_idx_seq = cut_idx
    pre_model = seq_model[:cut_idx_seq].eval()
    post_model = seq_model[cut_idx_seq:].eval()


    AS_model = AS_models[cut_idx_seq-1]
    AS_model.change_r(nAS) 
    Z_train, y_train = compute_Z_AS_space(
            AS_model, pre_model, post_model, train_loader, train_max_batch,device=device)


    print('Z_train, Z_test generated')
    mean = torch.mean(Z_train,0).to(device)
    var =  torch.std(Z_train,0).to(device)
    PCE_model = PCEModel(mean,var,nAS,p)

    coeff, training_score_LR, training_score_labels = PCE_model.Training(Z_train,
              y_train,train_labels[:Z_train.shape[0]])
    PCE_coeff = torch.tensor(coeff,dtype=torch.float32).to(device)
    print('Training score =', training_score_LR, 'label score', \
          training_score_labels)
 
    basis_layer = BasisLayer(PCE_model,device=device)
    ASNet = ASNET(pre_model, AS_model, basis_layer,PCE_coeff).to(device)

    print('Training and testing error for ASNet without re-training')
    
     
    optimizer = torch.optim.Adam([{'params': ASNet.premodel.parameters(), 
                                   'lr': 1e-3},
                                {'params': ASNet.AS.parameters(), 'lr': 1e-5},
                                {'params': ASNet.PCE.parameters(), 'lr': 1e-5}])
 
 
    epochs = 50   
    filename = './results/CIFAR100_ResNet110_ASNet'+\
            '_epoch_%d_nAS_%d_cutID_%d.pth'%(epochs,nAS,cut_idx)

    if os.path.isfile(filename): 
        [ASNet_pretrained,train_loss,test_loss] = torch.load(filename, map_location=device)
        ASNet.load_state_dict(ASNet_pretrained)
        print('ASNet trained {} epoches is loaded'.format(epochs))
    else:
        train_loss = []
        test_loss = []
        train_loss.append(compute_loss(ASNet, device, train_loader,topk=[1,5]))
        test_loss.append(compute_loss(ASNet, device, test_loader,topk=[1,5]))
        
        for epoch in range(1, epochs + 1):
            print('EPOCH {}'.format(epoch))
            train_loss.append(train_kd(ASNet, model, device, train_loader, optimizer, train_max_batch, 
                     alpha=0.1, temperature=1., epoch=epoch))
            test_loss.append(compute_loss(ASNet, device, test_loader,topk=[1,5]))

        torch.save([ASNet.state_dict(),train_loss,test_loss],filename)

    
    
    epochs_sparse = 100 
    print_every = 10
    lmd = 0.04
    lmd2 = 0.04
    optimizer = torch.optim.SGD([{'params': ASNet.premodel.parameters(), 'lr': 1e-4},
                            {'params': ASNet.AS.parameters(), 'lr': 1e-5},
                            {'params': ASNet.PCE.parameters(), 'lr': 1e-5}])
        
    ASNet_storage = torch.zeros([epochs_sparse+1,3])
    ASNet_flops = torch.zeros([epochs_sparse+1,3])
    
    ASNet_storage[0,0],ASNet_storage[0,1],ASNet_storage[0,2] = [Total_param(ASNet.premodel),
                                        Total_param(ASNet.AS), Total_param(ASNet.PCE)]
 
    ASNet_flops[0,0],ASNet_flops[0,1],ASNet_flops[0,2] = [Total_flops(ASNet.premodel,device),
                                        Total_flops(ASNet.AS,device), Total_flops(ASNet.PCE,device)]
    print('   Pre nnz = {:.2f}, AS nnz={:.2f}, PCE nnz={:.2f}'.format(
            ASNet_storage[0,0],ASNet_storage[0,1],ASNet_storage[0,2])) 
    
    for epoch in range(1, epochs_sparse + 1): 
        print('EPOCH: {}'.format(epochs+epoch))
        train_loss.append(train_l1(ASNet, device, train_loader, optimizer, 
                              train_max_batch,lmd,lmd2,lr_decrease=None,epoch=epoch))
        test_loss.append(compute_loss(ASNet, device, test_loader,topk=[1,5]))
        
        ASNet_storage[epoch,0] = Total_param_sparse(ASNet.premodel)
        ASNet_storage[epoch,1] = Total_param_sparse(ASNet.AS)
        ASNet_storage[epoch,2] = Total_param_sparse(ASNet.PCE)

        ASNet_flops[epoch,0],ASNet_flops[epoch,1],ASNet_flops[epoch,2] = [
            Total_flops_sparse(ASNet.premodel,device),
            Total_flops_sparse(ASNet.AS,device), Total_flops_sparse(ASNet.PCE,device)]
        
        print('   Pre nnz = {:.2f}/{:.2f}, AS nnz={:.2f}/{:.2f}, PCE nnz={:.2f}/{:.2f}'.format(
            ASNet_storage[epoch,0],ASNet_storage[0,0],ASNet_storage[epoch,1],
            ASNet_storage[0,1],ASNet_storage[epoch,2],ASNet_storage[0,2])) 

        if epoch%print_every ==0 or epoch==epochs_sparse:
            filename = './results/CIFAR100_ResNet110_ASNet_sparse'+\
                        '_epoch_%d_nAS_%d_cutID_%d.pth'%(epochs_sparse,nAS,cut_idx)
            torch.save([ASNet.state_dict(),train_loss,test_loss,ASNet_storage,ASNet_flops,lmd,lmd2],filename)

