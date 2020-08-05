#import numpy as np
#from scipy.linalg import svd as scipy_svd
#from scipy.sparse.linalg import svds as scipy_svds
from torch.autograd.gradcheck import zero_gradients, gradcheck
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.autograd import Variable
from collections import defaultdict
from functools import partial
#from copy import deepcopy
#import pdb

minrank = 400 
# The AS layer as given in Step 1 of Algorithm 3.1
# Algorithm 3.1 The training procedure of the active subspace network (ASNet)
class ASModel(nn.Module):
    def __init__(self, V,r,device):
        '''
        V here is n_features × r
        '''
        super(ASModel, self).__init__()
        self.device = device
        self.r_max = V.shape[1]
        self.r = self.r_max
        self.V_full = V.to(device)
        self.V = self.V_full[:,:r].to(device)
    def change_r(self, r_new):
        if r_new > self.r:
            raise ValueError('New AS dim is greater than the maximum!')
        self.r = r_new
        self.V = self.V_full[:,:r_new]#.to(self.device)
    def forward(self, x, r=None):
        x = x.view(x.size(0), -1)#.to(self.device)
        x = x @ self.V
        return x
    

## Computing the active subspace projection matrix and the eigen-distribution
## for small dataset
def get_AS_transform_input_smalldataset(model, dataset, dataset_labels,
                     r_max, attack_target = None, noise=None, loss=None, device='cpu'):
    
    if hasattr(model, 'features'): 
        for layer in model.features.children():
            if 'ReLU' in str(layer):
                layer.inplace = False
    else:
        for layer in model.children():
            if 'ReLU' in str(layer):
                layer.inplace = False
    
    model = model.to(device)
    model.eval()
#     model.cuda()
    dataset = dataset.to(device)
    noise = noise.to(device) 
    
    if noise is not None:
        samples = (dataset+noise).requires_grad_(True)
    else:
        samples = dataset.requires_grad_(True)
 
    res = model(samples)
    res = res.to(device)
    if len(dataset_labels)==1:
        dataset_labels = dataset_labels.repeat(res.shape[0])
    if loss is not None:
        if attack_target is None:
            res = loss(res, dataset_labels.to(device))#, reduction='none'
        else:
            res = loss(res, attack_target.repeat(res.shape[0].to(device)))
        res = res.view(-1, 1)  
    G = compute_grad_matrix(samples, res)
    U, Sigma, V = randomized_svd(G, r_max)
    ASlayer = ASModel(V.t(),r_max,device)
    return ASlayer, Sigma


# The randomized SVD algorithm
# It is faster than SVD in general
def randomized_svd(A, k):
    '''
    Computes randomized rank-k truncated SVD of matrix A:
    A ≈ U @ torch.diag(Sigma) @ V.t()
    Args:
        A: input torch.Tensor matrix
        k: rank
    Returns:
        U, Sigma, V
    '''
    device = A.device
    with torch.no_grad():
        m, n = A.shape
        Omega = torch.randn(n, k, device=device)
        Y = A @ Omega
        Q, R = torch.qr(Y)
        B = Q.t() @ A
        Uhat, Sigma, V = torch.svd(B)
        U = Q @ Uhat
        return U, Sigma, V.t()

    
## Algorithm 3.2: The frequent direction algorithm for computing the active subspace
def get_ASModel_FD(model, train_loader, cut_layer, max_batch,r_max,device): 

    as_emb = streamASEmbedding(model, train_loader,device,max_batch)
    as_emb.forward_backward(cut_layer)
    ASlayers = defaultdict(list)
    Sigmas = defaultdict(list)
    for key in as_emb.fds.keys():
        #print('Starting to Build Layer:',key)
        s, Vt = as_emb.fds[key].get()
        ASlayers[key] = ASModel(Vt.t(),r_max,device) 
        Sigmas[key] = s
        print('Finished Building AS model for layer:',key)
    return ASlayers, Sigmas 

##  subroutine used in get_ASModel_FD()
class streamASEmbedding():
    def __init__(self, model, loader, device,batch_count = 1):
        self.model = model
        self.loader = loader
        self.device = device
        self.batch_count = batch_count
        self.logg_every = 2
        
        self.model.to(self.device)
        self.activations = defaultdict(list)
        self.fds = defaultdict(list)
        
        self.d_rate = 0.8

    def forward_backward(self,names):
        self.model.train()      
        if type(names)!=list:
            m = self.model[names-1]
            m.register_backward_hook(partial(self.save_activation, names-1))
        else:
            for i, layer in enumerate(names): 
                m = self.model[layer-1]
                m.register_backward_hook(partial(self.save_activation, layer-1))
#         for name, m in self.model.named_modules():
#                 # if type(m)==nn.Linear or type(m) == nn.Conv2d:
#                     # partial to assign the layer name to each hook
#                     m.register_backward_hook(partial(self.save_activation, name))
                
        for batch_idx, (data, target) in enumerate(self.loader):
            
            data, target = data.to(self.device), target.to(self.device)
                
            #data, target = Variable(data), Variable(target)
 
            output = self.model(data)
        
            loss = F.nll_loss(output, target)
            loss.backward()
            
            ### Update Frequent directions 
            self.fd_step()
                       
            if batch_idx >= self.batch_count:
                break   
                
    def fd_step(self):
        for key in self.activations:
            if key not in self.fds.keys():
                bs, n_features = torch.cat(self.activations[key], 0).shape
                
                self.fds[key] = FrequentDirections(n = n_features,\
                       d = min(minrank,int(n_features*self.d_rate)))
            else:
                for row in torch.cat(self.activations[key], 0):
                    self.fds[key].append(row)
  

    
    def save_activation(self, name, mod, grad_inp, grad_out):
#         if type(mod) == nn.Conv2d:
#             grad_x, grad_weight, grad_bias = grad_inp
#         elif type(mod) == nn.Linear:
#             grad_bias, grad_x, grad_weight = grad_inp
#             grad_weight = grad_weight.t().contiguous()

        bs = grad_out[0].shape[0]
        self.activations[name] = [grad_out[0].view(bs, -1).cpu()]
#         self.activations[name].append(grad_out[0].view(bs, -1).cpu())
        



# Subroutine used in streamASEmbedding.fd_step()
class FrequentDirections():

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.m = 2 * self.d
        self._sketch = torch.zeros((self.m, self.n), dtype=torch.float32)

        self.nextZeroRow = 0

    def append(self, vector):
        if torch.nonzero(vector).size(0) == 0:
            return

        if self.nextZeroRow >= self.m:
            self.__rotate__()

        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1

    def __rotate__(self):
        Vt,s,_ = randomized_svd(self._sketch.t(), self.d)
        Vt = Vt.t()
        sShrunk = torch.sqrt(s[:self.d]**2 - s[self.d - 1]**2)
        self._sketch[:self.d:, :] = torch.diag(sShrunk) @ Vt[:self.d, :]
        self._sketch[self.d:, :] = 0
        self.nextZeroRow = self.d

    def get(self, rotate=True, take_root=False):
        if rotate:
            Vt,s,_ = randomized_svd(self._sketch.t(), self.d)
            Vt = Vt.t()
            if take_root:
                return torch.diag(torch.sqrt(s[:self.d])) @ Vt[:self.d, :]
            else:
#                 return np.diag(s[:self.d]) @ Vt[:self.d, :]
                return s[:self.d], Vt[:self.d, :]


        return self._sketch[:self.d, :]
    

## for computing partial fx
def compute_grad_matrix(x, fx):
    assert x.requires_grad
    num_classes = fx.shape[1]
    jacobian = torch.zeros(num_classes, *x.shape, device=x.device)
    grad_output = torch.zeros(*fx.shape, device=x.device)
 
    for i in range(num_classes):
        zero_gradients(x)
        grad_output.zero_()
        grad_output[:, i] = 1
        fx.backward(grad_output, retain_graph=True)
  
        jacobian[i] = x.grad.data
    jacobian = jacobian.transpose(0, 1).contiguous()
    # (n_samples × n_classes) × n_features
    # n_outputs = jacobian.shape[1]
    return jacobian.view(jacobian.shape[0] * jacobian.shape[1], -1)


 
