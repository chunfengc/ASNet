import torch
import numpy as np
import torch.nn as nn
import pdb
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
 
def Total_param_sparse(ASNet,stoarge_per_param=4):
    nnz = 0
    for name, m in ASNet.named_modules():
        if type(m) == nn.Conv2d or type(m)==nn.Linear:  
            nnz += m.weight.data.nonzero().shape[0]
    return nnz/2**20* stoarge_per_param   

def Total_param(model,stoarge_per_param=4):
    total_params = 0 
    for t in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(t.data.cpu().numpy().shape)
    return total_params/2**20* stoarge_per_param   

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

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
        return
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
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
 
def PossibleCutIdx(seq_model):
    cutidx = []
    for i, m in seq_model.named_modules():
        if type(m)==nn.Linear or type(m) == nn.Conv2d:
            cutidx.append(int(i)) #  Find the Linear or Conv2d Layer Idx
    return cutidx 

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
 
def active_eigs(sigma,delta=0.95): 
    nm_sigm = sum(sigma**2) 
    nm = 0
    for i in range(len(sigma)):
        nm += sigma[i]**2
        if nm> nm_sigm*delta**2:
            break
    return i 


# def storage_PCE(nAS,constant_storage = 4,nclass=10,porder=2):
#     if porder==2:
#         return constant_storage*nclass*(nAS+porder)*(nAS+porder-1)/2  
    
# def storage_AS(nAS,nfeature,constant_storage = 4):
#     return constant_storage*nfeature*nAS 

# def compute_label_error(pred_labels, labels, device):
#     if type(pred_labels)== np.ndarray:
#         pred_labels = torch.from_numpy(pred_labels) 
#     if type(labels)== np.ndarray:
#         labels = torch.from_numpy(pred_labels)
#     correct = labels.to(device).eq(pred_labels.to(device)).sum()
#     return correct.double()/len(labels)



# def randomized_svd(A, k=200):
#     device = A.device
#     with torch.no_grad():
#         m, n = A.shape
#         Omega = torch.randn(n, k, device=device)
#         Y = A @ Omega
#         Q, R = torch.qr(Y)
#         B = Q.t() @ A
#         Uhat, Sigma, V = torch.svd(B)
#         U = Q @ Uhat
#         return U, Sigma, V.t()
    
# def compute_AS_dist(image_AS, tmp_AS, Sigma):
#     AS_num = 50
#     dist = ((image_AS[:AS_num]-tmp_AS[:AS_num])**2).dot(Sigma[:AS_num])
#     return dist**(0.5)

# def get_seq_model_vgg(model):
#     res = nn.Sequential(*(list(model.features.children()) + 
#            [model.avgpool]+[Vectorize()] +
#           list(model.classifier.children())))
    
#     return res


# def compute_intermediate_feature(pre_model,data_loader,
#                   num_batch,device):
#     X_out = torch.zeros(0).cpu()
#     for idx_, (batch,target) in enumerate(data_loader):
#         if idx_>=num_batch:
#             break 
#         batch = batch.to(device)
#         with torch.no_grad():
#             x = pre_model(batch)
            
#         X_out = torch.cat([X_out, x.view([batch.shape[0],-1]).cpu()])
        
#     return X_out.cpu()

            
# def compute_Z_AS_PCA(AS_model, pre_model, post_model, data_loader, \
#                      num_batch,device,U=[]):
#     X_train = torch.zeros(0)
#     y_train = torch.zeros(0)
#     Z_PCA = torch.zeros(0)
#     for idx_, (batch,target) in enumerate(data_loader):
#         if idx_>=num_batch:
#             break
            
#         batch = batch.to(device)
#         with torch.no_grad():
#             x = pre_model(batch)
#             as_data = AS_model(x).cpu()
#             y_data = post_model(x).cpu()
#         if len(U)==0:
#             Z_PCA = torch.cat([Z_PCA,x.cpu().view(batch.shape[0],-1)])
#         else:
#             Z_PCA = torch.cat([Z_PCA,x.cpu().view(batch.shape[0],-1)@U])
#         X_train = torch.cat([X_train, as_data])
#         y_train = torch.cat([y_train, y_data])
        
#     if len(U)==0:
#         U,_,_ = randomized_svd(Z_PCA.t(),as_data.shape[1])
#         Z_PCA = Z_PCA @ U    
#         return X_train, Z_PCA, y_train,U
#     else:
#         return X_train, Z_PCA, y_train
    

# def get_seq_model_resnet_old(model):
#     seq = nn.Sequential(*(list([model.module.conv1]+[model.module.bn1]+[nn.ReLU()]+\
#                   list(model.module.layer1.children())+
#                      list(model.module.layer2.children())+
#                      list(model.module.layer3.children())+\
#                     [nn.AvgPool2d(8)]+[Vectorize()]+[model.module.linear])))
#     return seq

# def get_seq_model_resnet_2(model):

    
#     seq = nn.Sequential(
#             nn.Sequential(model.module.conv1,
#                           model.module.bn1,
#                           nn.ReLU()),
#             *list(list(model.module.layer1.children())+
#                      list(model.module.layer2.children())+
#                      list(model.module.layer3.children())),
#             nn.Sequential(nn.AvgPool2d(8),Vectorize(),model.module.fc))
#     return seq

# def Num_ASnet_flops(ASNet,device,p=2,nAS=50,is_count_premodel=True):
#     flops = 0.0
    
#         flops += Num_flops(ASNet.premodel,device)
        
#     flops += ASNet.AS.weight.data.nonzero().shape[0]#ASNet.AS.in_features*ASNet.AS.out_features
#     flops += ASNet.PCE.weight.data.nonzero().shape[0]# ASNet.PCE.in_features*ASNet.PCE.out_features
#     if is_count_premodel:
#         flops += p*(ASNet.PCE.in_features + nAS)#Basis function
    
#     return float(flops)

# def ACC_add_noise_input(model,noise,noise_level,data_loader,device):
#     train_correct = 0.
#     noise = noise.to(device)*noise_level
#     for (idx,(image,label)) in enumerate(data_loader): 
#         image = image.to(device)
#         output = model(image + noise)
#         pred = output.max(1, keepdim=True)[1] 
#         train_correct += pred.eq(
#             label.to(device).view_as(pred)).sum().item() 

#     return train_correct/len(data_loader.dataset)*100

# def backprop_noise_output(x0,model,noise_out,image_data,max_iter = 1000):
#     loss = F.mse_loss
#     x = Variable(x0.clone(), requires_grad=True)
#     eps = 1e-4
#     optimizer = torch.optim.Adagrad([x], lr=1e-2)
#     m = copy.deepcopy(model)
#     for i in range(max_iter):
#         optimizer.zero_grad()
#         res = loss(m(x),noise_out)
#         res.backward(retain_graph=True )
#         optimizer.step()
#          #x = (x-sz*x.grad.data).clone().detach().requires_grad_(True)
#         if res<eps: 
#             break
#     return x

# def ACC_add_noise_cutidx(pre_model,post_model,noise,noise_level,data_loader):
#     train_correct = 0.
#     noise_here = noise * noise_level
#     for (idx,(image,label)) in enumerate(data_loader): 
#         x1 = pre_model(image.to(device)) 
#         output = post_model(x1+noise_here) 
#         pred = output.max(1, keepdim=True)[1] 
#         train_correct += pred.eq(
#             label.to(device).view_as(pred)).sum().item() 

#     return train_correct/len(data_loader.dataset)*100

# def compute_min_attack_radius(noise, image_data,image_label, model):
#     level_lb = 0.0
#     ub_max = 1e3
#     level_ub = ub_max
#     level = 1.0
#     while abs(level_lb-level_ub)>1e-2 and level < level_ub: 
#         pred_idx = model(noise*level + image_data).argmax()
#         if pred_idx == image_label: 
#             level_lb = level
#             if level_ub == ub_max:
#                 level += 0.5
#             else:
#                 level = (level_lb + level_ub)/2
#         else:
#             level_ub = level
#             level = (level_lb + level_ub)/2
            
#         #print(level_lb,level,level_ub,pred_idx)
#     return level


# def compute_topk_loss(model, device, test_loader,topk=[1,5]):
#     model.eval()
#     model.to(device)
#     test_loss = 0
    
#     res = []
#     maxk = max(topk)
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
#             pred = pred.t()
#             correct = pred.eq(target.view(1, -1).expand_as(pred))
#             for k in topk:
#                 correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#                 res.append(correct_k)
#     test_loss /= len(test_loader.dataset) 
#     correct = torch.FloatTensor(res).view(-1,len(topk)).sum(dim=0)
#     test_accuracy = 100. * correct / len(test_loader.dataset)
#     for idx,k in enumerate(topk): 
#         print(' Top {}:  Accuracy: {}/{} ({:.2f}%)'.format(
#             k,  correct[idx], len(test_loader.dataset),
#             test_accuracy[idx]))
#     return test_accuracy

# def Flops_resnet_basic32(model,num_layer,device,is_module=True):
    
    
#     n = (num_layer-2)//6
    
#     xold = torch.ones([1,3,32,32]).to(device)
#     if is_module:
#         x = model.module.conv1(xold)
#     else:
#         x = model.conv1(xold)
#     flop = xold.shape[1]*x.shape[1:].numel()*9
    
#     xold = x
#     if is_module:
#         x = model.module.layer1(x).to(device)
#     else:
#         x = model.layer1(x).to(device)
#     flop += 2*n*xold.shape[1]*x.shape[1:].numel()*9
    
#     xold = x
#     if is_module:
#         x = model.module.layer2(x).to(device)
#     else:
#         x = model.layer1(x).to(device)
#     flop += 2*n*xold.shape[1]*x.shape[1:].numel()*9
    
#     xold = x
#     if is_module:
#         x = model.module.layer3(x).to(device)
#     else:
#         x = model.layer1(x).to(device)
#     flop += 2*n*xold.shape[1]*x.shape[1:].numel()*9
    
#     if is_module:
#         flop += model.module.linear.in_features*model.module.linear.out_features
#     else:
#         flop += model.linear.in_features*model.linear.out_features
#     return flop