import torch
#import torch.nn as nn
from .ASModel import *
#import pdb

# projection v to \|v\|\le xi
def proj_v(v, xi, proj_type='l2'):
    v2 = v.clone()
    if proj_type=='l0':
        xi = int(xi//1)
        v_shape = v.shape
        v2 = v2.view(-1)
        _, indices = torch.topk(abs(v2), xi)
        topk = v2[indices]
        v2 = torch.zeros(v.shape)
        v2[indices] = topk
        v2 = v2.view(v_shape)
        return v2
    else:
        return v2*min(1,xi/torch.norm(v2))

# This is a subalgorithm for AS_attack_input_small_dataset_sz
# the backtracking method to compute the stepsize sz and its sign
# given the previous perturbation vector v_AS and the update step dv = v_AS_i
def AS_attack_update_sz(model, dataset, dataset_labels, xi,v_AS, v_AS_i, attack_ratio_old,
                        proj_type='l2', device='cpu', 
                        szmax = 2, maxiter=10, decrease_cons = 0.5):
    
    model = model.to(device)
    dataset = dataset.to(device)
    dataset_labels = dataset_labels.to(device)
    v_AS_i = v_AS_i.to(device) 
    num_samples = dataset.shape[0]
    
    if len(dataset_labels)==1:
        dataset_labels = dataset_labels.repeat(num_samples)
       
    
    sz = szmax
    sz_best = None
    for i in range(maxiter): 
        v_AS_pos = proj_v(v_AS+sz*v_AS_i, xi, proj_type)
        output_post = model(dataset+v_AS_pos)
        unattacked_idx = torch.nonzero(output_post.argmax(dim=1)==dataset_labels)
        attack_ratio_pos = 1-len(unattacked_idx)/num_samples 
          
        v_AS_neg = proj_v(v_AS-sz*v_AS_i, xi, proj_type)
        output_neg = model(dataset+v_AS_neg)
        unattacked_idx_neg = torch.nonzero(output_neg.argmax(dim=1)==dataset_labels)
        attack_ratio_neg = 1-len(unattacked_idx_neg)/num_samples 
       
        if attack_ratio_pos > attack_ratio_neg:
            v_AS_best = v_AS_pos
            attack_ratio_best, unattacked_idx = attack_ratio_pos, unattacked_idx.view(-1) 
            if attack_ratio_pos > attack_ratio_old: 
                sz_best = sz
                break
        else:
            v_AS_best = v_AS_neg
            attack_ratio_best, unattacked_idx = attack_ratio_neg, unattacked_idx_neg.view(-1)
            if attack_ratio_neg > attack_ratio_old:
                sz_best = -sz 
                break
        sz*=decrease_cons
        
    return v_AS_best, attack_ratio_best, unattacked_idx, sz_best


# The Recursive Active Subspace for Universal Attack in Algorithm 4.1
# We call two subalgorithms: 
    # 1. get_AS_transform_input_smalldataset() to compute the dominant AS direction
    # 2. AS_attack_update_sz() to obtain the stepsize
def AS_attack_input_small_dataset_sz(model,dataset, dataset_labels, 
                              noise_level, loss, proj_type='l2',max_iter = 100, 
                              attack_target = None, r_max=1,device='cpu',szmax=None,
                              no_progress_stop=10):
    
    v_AS = torch.zeros([1,*dataset.shape[1:]]).to(device) 
    dataset = dataset.to(device)
    if len(dataset_labels)==1:
        dataset_labels = dataset_labels.to(device).repeat(dataset.shape[0])
    attack_ratio,attack_ratio_best = -1,-1
    no_progress_iter = 0
    if szmax is None:
        szmax = noise_level*noise_level
        
    for itr in range(max_iter):   
        if itr > 0:
            ASlayer, Sigma = get_AS_transform_input_smalldataset(model, 
                         dataset[unattacked_idx], dataset_labels[unattacked_idx],
                         r_max, noise=v_AS,attack_target=attack_target, loss=loss, device=device)
        else:
            ASlayer, Sigma = get_AS_transform_input_smalldataset(model,  dataset, dataset_labels, 
                         r_max, noise=v_AS,attack_target=attack_target, loss=loss, device=device)
 
 
        v_AS_i = ASlayer.V[:,0].reshape(v_AS.shape).to(device) 
        v_AS, attack_ratio, unattacked_idx, sz = AS_attack_update_sz(model, 
                          dataset, dataset_labels, noise_level,v_AS, v_AS_i, attack_ratio,
                          proj_type=proj_type, device=device,szmax=szmax)
            
        if attack_ratio > attack_ratio_best:
            attack_ratio_best = attack_ratio
            v_AS_best = v_AS.clone()
            no_progress_iter = 0
        else:
            no_progress_iter += 1
        
        
        if no_progress_iter>no_progress_stop or len(unattacked_idx)==0:
            break
            
    print('   AS iteration {}/{}, attack ratio {:.2f},|v|={:.2f}'.format(
            itr, max_iter, attack_ratio, torch.norm(v_AS)))

    return v_AS_best/(1e-10+torch.norm(v_AS_best))

