import torch
import numpy as np
import torch.nn as nn
import pdb
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
 
# The total number of parameters
def Total_param(model,stoarge_per_param=4):
    total_params = 0 
    for t in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(t.data.cpu().numpy().shape)
    return total_params/2**20* stoarge_per_param   

# The total number of nonzero parameters
def Total_param_sparse(ASNet,stoarge_per_param=4):
    nnz = 0
    for name, m in ASNet.named_modules():
        if type(m) == nn.Conv2d or type(m)==nn.Linear:  
            nnz += m.weight.data.nonzero().shape[0]
    return nnz/2**20* stoarge_per_param   

# The total number of flops for nonzero elements only
def Total_flops_sparse(model,device,is_ASNet=False,p=2,nAS=50): 
    x = torch.ones([1,3,32,32]).to(device) 
    flops = 0.
    for i, m in model.named_modules():
        xold = x
        if type(m) == nn.MaxPool2d:
            x = m(x)
        if type(m) == nn.Conv2d:
            x = m(x)  
            flops += x.shape[2:].numel()*m.weight.data.nonzero().shape[0]
        if type(m) == nn.Linear:
            flops += m.weight.data.nonzero().shape[0]
    if is_ASNet:
        flops += p*(model.PCE.in_features + nAS)#Basis function
    return float(flops)/10**6

# The total number of flops
def Total_flops(model,device,is_ASNet=False,p=2,nAS=50): 
    x = torch.ones([1,3,32,32]).to(device) 
    flops = 0.
    for i, m in model.named_modules():
        xold = x
        if type(m) == nn.MaxPool2d:
            x = m(x)
            #flops += x.shape[1:].numel()
        if type(m) == nn.Conv2d:
            x = m(x) 
            flops += xold.shape[1]*x.shape[1:].numel()*\
                    torch.tensor(m.kernel_size).prod()
        if type(m) == nn.Linear:
            flops += m.in_features*m.out_features
            
    if is_ASNet:
        flops += p*(model.PCE.in_features + nAS)#Basis function
    return float(flops)/10**6

# To reshape the tensor into a vector
class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
        return
    def forward(self, x):
        return x.view(x.shape[0], -1)
  
# To get the sequential formula of the VGG model
def get_seq_model(model):
    '''
    Takes a model with model.features and model.classifier and
    returns a sequential model
    '''
    if list(model.classifier.children()):
        res = nn.Sequential(*(list(model.features.children()) + 
              [Vectorize()] + 
              list(model.classifier.children())))
    else:
        res = nn.Sequential(*(list(model.features.children()) + 
              [Vectorize()] + 
              [model.classifier])) 
    return res 


def get_seq_model_resnet_nomodule(model): 
    seq = nn.Sequential(
            nn.Sequential(model.conv1,model.bn1,nn.ReLU()),
            *list(list(model.layer1.children())+
                     list(model.layer2.children())+
                     list(model.layer3.children())),
            nn.Sequential(nn.AvgPool2d(8),Vectorize(),model.linear))
    return seq

def get_seq_model_resnet(model):  
    seq = nn.Sequential(
            nn.Sequential(model.module.conv1,
                          model.module.bn1,
                          nn.ReLU()),
            *list(list(model.module.layer1.children())+
                     list(model.module.layer2.children())+
                     list(model.module.layer3.children())),
            nn.Sequential(nn.AvgPool2d(8),Vectorize(),model.module.linear))
    return seq
 
## We only cut the network at the Conv or FC layer
def PossibleCutIdx(seq_model):
    cutidx = []
    for i, m in seq_model.named_modules():
        if type(m)==nn.Linear or type(m) == nn.Conv2d:
            cutidx.append(int(i)) #  Find the Linear or Conv2d Layer Idx
    return cutidx 

## To construct the dataset D for obtaining the active subspace direction
def compute_Z_AS_space(AS_model, pre_model, post_model, data_loader, num_batch=10,device='cpu'):
    
    X_train = torch.zeros(0)
    y_train = torch.zeros(0)
    for idx_, (batch,target) in enumerate(data_loader):
        if idx_>=num_batch:
            break
            
        batch = batch.to(device)
        with torch.no_grad():
            x = pre_model(batch)
            as_data = AS_model(x).cpu()
            y_data = post_model(x).cpu()
        X_train = torch.cat([X_train, as_data.cpu()])
        y_train = torch.cat([y_train, y_data.cpu()])
        
    return X_train, y_train 
 
# Definition 3.1: the number of active neurons
def active_eigs(sigma,delta=0.95): 
    nm_sigm = sum(sigma**2) 
    nm = 0
    for i in range(len(sigma)):
        nm += sigma[i]**2
        if nm> nm_sigm*delta**2:
            break
    return i 


