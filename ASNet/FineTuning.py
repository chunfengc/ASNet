import torch
#from IPython import display
import torch.nn.functional as F
import torch.nn as nn
#import matplotlib.pyplot as plt 

#import pdb


##  compute the top-k accuracy of model for dataset=test_loader
def compute_loss(model, device, test_loader,is_print=True,topk=[1]):    
    model.eval()
    model.to(device)
    test_loss = 0
    
    res = []
    maxk = max(topk)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
    test_loss /= len(test_loader.dataset) 
    correct = torch.FloatTensor(res).view(-1,len(topk)).sum(dim=0)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    for idx,k in enumerate(topk): 
        print(' Top {}:  Accuracy: {}/{} ({:.2f}%)'.format(
            k,  correct[idx], len(test_loader.dataset),
            test_accuracy[idx]))
    if len(topk)==1:
        return test_accuracy[0]
    else:
        return test_accuracy

## The soft-thresholding for l1 regularized proximal gradient
def soft_threshold(x,lmd=0.1): 
    x = x*torch.relu(1-lmd/(1e-30+torch.abs(x))) 
    return x

## To retrain the model on the dataset=train_loader
def train(model, device, train_loader, optimizer,lr_decrease=None,epoch=1):
    model.train().to(device)
    correct = 0.0
     
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward() 
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    accuracy = correct / len(train_loader.dataset) * 100.0
    if lr_decrease is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decrease 
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (epoch)/(epoch+1) 
    return accuracy

## retraining the model with l1 regularization
## lmd is the parameter for the AS layer and the PCE layer
## lmd2 is the parameter for the premodel
## when both lmd and lmd2 equals to zero, then this algorithm reduced to train()
    
def train_l1(model, device, train_loader, optimizer,train_max_batch,lmd = 0.1,lmd2=0.1,
            lr_decrease=None,epoch=1):
    model.train().to(device)
    correct = 0.0
     
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward() 
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        for name, m in model.premodel.named_modules():
            if type(m) == nn.Conv2d:
                m.weight.data = soft_threshold(m.weight.data,lr*lmd2) 

        lr = optimizer.param_groups[1]['lr']
        model.AS.weight.data = soft_threshold(\
                model.AS.weight.data, lmd*lr)

        lr = optimizer.param_groups[2]['lr']
        model.PCE.weight.data = soft_threshold(\
            model.PCE.weight.data, lmd*lr)

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()


        if batch_idx >= train_max_batch:
            break
    accuracy = correct / len(train_loader.dataset) * 100.0
    
    if lr_decrease is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decrease
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (epoch)/(epoch+1)
    return accuracy

## retraining the model with knowledge distillation
## when alpha=0, it reduced to the original training
def train_kd(student, teacher, device, train_loader, optimizer, train_max_batch, 
             alpha=0.0, temperature=1.,lr_decrease=None, epoch=1):
    student.train()
    teacher.eval()
    student.to(device)
    teacher.to(device) 
    correct = 0.0  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = student(data)
        output_teacher = teacher(data)
   
        loss = nn.KLDivLoss()(F.log_softmax(output / temperature, dim=1),F.softmax(output_teacher / temperature, dim=1)
                             )*(alpha*temperature*temperature) + \
                 F.cross_entropy(output, target) * (1. - alpha)
   
        loss.backward() 
        optimizer.step()
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    accuracy = correct / len(train_loader.dataset) * 100.0
    if lr_decrease is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decrease
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*(epoch)/(epoch+1)
    return accuracy


 
