import torch
import torch.nn as nn
from .ASModel import *
import pdb 

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
        
    return v_AS_best,attack_ratio_best, unattacked_idx, sz_best

    
def AS_attack_input_small_dataset_sz(model,dataset, dataset_labels, 
                              noise_level, loss, proj_type='l2',max_iter = 100, 
                              attack_target = None, r_max=1,device='cpu',szmax=None,
                              no_progress_stop=10):
    
    v_AS = torch.zeros([1,*dataset.shape[1:]]).to(device)
    sign_v = 1
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




# def compute_attack_ratio_smalldataset(model, dataset, dataset_labels, noise,loss=None,
#                                      attack_target=None, max_batch=1, device='cpu'):
     

#     model = model.to(device)
#     dataset = dataset.to(device)
#     dataset_labels = dataset_labels.to(device)
#     noise = noise.to(device)
#     output_post = model(dataset+noise) 
#     num_samples = dataset.shape[0]

#     if len(dataset_labels)==1:
#         dataset_labels = dataset_labels.repeat(num_samples)
#     if attack_target is None: 
#         unattacked_idx = torch.nonzero(output_post.argmax(dim=1)==dataset_labels)
#         attack_ratio = len(unattacked_idx) 
#     else: 
#         unattacked_idx = torch.nonzero(output_post.argmax(dim=1)!=attack_target)
#         attack_ratio = len(unattacked_idx)  
#     attack_ratio = 1- attack_ratio/num_samples

#     return attack_ratio,unattacked_idx.view(-1)

# def compute_loss_post_neg_smalldataset(model,dataset,dataset_labels, 
#                      noise,loss,attack_target=None,device='cpu'):
 
#     model = model.to(device)
#     dataset = dataset.to(device)
#     dataset_labels = dataset_labels.to(device)
#     output_post = model(dataset+noise)
#     output_neg = model(dataset-noise)
#     num_samples = dataset.shape[0]
#     loss_post_neg = torch.zeros(2)
  
#     if len(dataset_labels)==1:
#         dataset_labels = dataset_labels.repeat(num_samples)
#     if attack_target is None:
#         loss_post_neg[0] = loss(output_post,dataset_labels)
#         loss_post_neg[1] = loss(output_neg,dataset_labels)
         
#     else:
#         loss_post_neg[0] = loss(output_post,attack_target.repeat(num_samples))
#         loss_post_neg[1] = loss(output_neg,attack_target.repeat(num_samples))
 
#     return loss_post_neg

# def AS_attack_input_small_dataset(model,dataset, dataset_labels, 
#                               noise_level, loss, proj_type='l2',max_iter = 100, 
#                               attack_target = None, r_max=1,device='cpu',const = 1):
    
#     v_AS = torch.zeros([1,*dataset.shape[1:]]).to(device)
#     v_AS_best = torch.zeros([1,*dataset.shape[1:]]).to(device)
#     sign_v = 1
#     dataset = dataset.to(device)
#     if len(dataset_labels)==1:
#         dataset_labels = dataset_labels.to(device).repeat(dataset.shape[0])
#     attack_ratio = 0
#     attack_ratio_best = 0
#     no_progress_iter = 0
#     for itr in range(max_iter):  
 
#         if itr > 0:
#             ASlayer, Sigma = get_AS_transform_input_smalldataset(model, 
#                          dataset[unattacked_idx], dataset_labels[unattacked_idx],
#                          r_max, noise=v_AS,attack_target=attack_target, loss=loss, device=device)
#         else:
#             ASlayer, Sigma = get_AS_transform_input_smalldataset(model,  dataset, dataset_labels, 
#                          r_max, noise=v_AS,attack_target=attack_target, loss=loss, device=device)
 
 
#         v_AS_i = ASlayer.V[:,0].reshape(v_AS.shape).to(device)
#         v_AS_i = proj_v(v_AS_i, noise_level, proj_type)
        
        
#         loss_post_neg = compute_loss_post_neg_smalldataset(model,
#                              dataset, dataset_labels, v_AS_i, loss, 
#                               attack_target,device=device)
        
#         if attack_target is not None:# to minimize loss 
#             if loss_post_neg[0]>loss_post_neg[1]: 
#                 sign_v = -1
#             else:
#                 sign_v = 1
#         else:
#             if loss_post_neg[0]>loss_post_neg[1]:
#                 sign_v = 1 
#             else:
#                 sign_v = -1
                    
#         v_AS += const*sign_v*v_AS_i
#         v_AS = proj_v(v_AS, noise_level, proj_type)
#         attack_ratio_old = attack_ratio
#         attack_ratio, unattacked_idx = compute_attack_ratio_smalldataset(model, 
#                                             dataset, dataset_labels, v_AS,loss)
        
#         if attack_ratio < attack_ratio_old:
#             no_progress_iter+=1
#         else: 
#             no_progress_iter = 0
            
#         if attack_ratio > attack_ratio_best:
#             attack_ratio_best = attack_ratio
#             v_AS_best = v_AS.clone()
#         if attack_ratio_best>0.4 and (no_progress_iter>=3 or len(unattacked_idx)==0):
#             break   
#     print('   AS iteration {}/{}, attack ratio {:.2f},|v|={:.2f}'.format(
#             itr,max_iter,attack_ratio,torch.norm(v_AS)))
#     return v_AS_best/(1e-10+torch.norm(v_AS_best))


# def compute_attack_ratio(model,dataloader,max_samples, noise,loss,
#                      is_small_dataset=False, attack_target=None,max_batch=1,device='cpu'):
     
#     attack_ratio = 0.0
 
#     if is_small_dataset:
#         model = model.to(device)
#         dataset = dataloader.to(device)
#         dataset_labels = max_samples.to(device)
#         output_post = model(dataset+noise) 
#         num_samples = dataset.shape[0]

#         if len(dataset_labels)==1:
#             dataset_labels = dataset_labels.repeat(num_samples)
#         if attack_target is None: 
#             attack_ratio += torch.sum(output_post.argmax(dim=1)!=dataset_labels) 
#         else: 
#             attack_ratio += torch.sum(output_post.argmax(dim=1)==attack_target) 
#         attack_ratio= attack_ratio.double()/num_samples
        
#     else:
#         num_samples = 0
#         for ib, (image,target) in enumerate(dataloader): 
#             output_post = model(image+noise)

#             if attack_target is None: 
#                 attack_ratio += torch.sum(output_post.argmax(dim=1)!=target) 
#             else:
#                 attack_ratio += torch.sum(output_post.argmax(dim=1)==attack_target) 

#             num_samples += image.shape[0]
#             if ib+1 >= max_batch:
#                 break

#         attack_ratio= attack_ratio.double()/num_samples
#     return attack_ratio

# def compute_loss_post_neg(model,dataloader,max_samples, noise,loss,
#                      is_small_dataset=False, attack_target=None,max_batch=1,device='cpu'):
    
#     loss_post_neg = torch.zeros(2)

#     for ib, (image,target) in enumerate(dataloader): 
#         output_post = model(image+noise)
#         output_neg = model(image-noise)
        
#         if attack_target is None:
#             loss_post_neg[0] += loss(output_post,target)
#             loss_post_neg[1] += loss(output_neg,target)
            
#         else:
#             loss_post_neg[0] += loss(output_post,attack_target.repeat(output_post.shape[0]))
#             loss_post_neg[1] += loss(output_neg,attack_target.repeat(output_neg.shape[0]))

#         if ib+1 >= max_batch:
#             break
#     return loss_post_neg 

# def AS_attack_input(model,dataloader, max_samples, noise_level, loss, 
#                       is_small_dataset=False, rho = 0.8, max_iter = 100, 
#                       attack_target = None, r_max=1, device='cpu'):
    
#     if is_small_dataset==False:
#         v_AS = torch.zeros([1,*dataloader.dataset[0][0].shape]).to(device).to(device)
#     else:
#         v_AS = torch.zeros([1,*dataloader.shape[1:]]).to(device)
#     sign_v = 1 
#     if is_small_dataset:
#         dataloader = dataloader.to(device)
#         max_samples = max_samples.to(device)
#     for itr in range(max_iter):
 
#         if is_small_dataset: 
#             ASlayer, Sigma = get_AS_transform_input_smalldataset(model, dataloader, max_samples, 
#                          r_max, noise=v_AS,attack_target=attack_target, loss=loss, device=device)
#         else:
#             ASlayer, Sigma = get_AS_transform_input(model, dataloader, max_samples, 
#                          r_max, noise=v_AS,attack_target=attack_target, loss=loss, device=device)
#         v_AS_i = ASlayer.V[:,0].reshape(v_AS.shape).to(device)
 
#         if is_small_dataset == False:
#             loss_post_neg = compute_loss_post_neg(model,
#                              dataloader, max_samples, noise_level*v_AS_i, loss, 
#                              attack_target, max_batch=1, device=device)
#         else:
#             loss_post_neg = compute_loss_post_neg_smalldataset(model,
#                              dataloader, max_samples, noise_level*v_AS_i, loss, 
#                               attack_target,device=device)
        
#         if attack_target is not None:# to minimize loss 
#             if loss_post_neg[0]>loss_post_neg[1]: 
#                 sign_v = -1
#             else:
#                 sign_v = 1
#         else:
#             if loss_post_neg[0]>loss_post_neg[1]:
#                 sign_v = 1 
#             else:
#                 sign_v = -1
                    
#         v_AS += sign_v*v_AS_i
#         v_AS = proj_v(v_AS, noise_level)
        
#         attack_ratio = compute_attack_ratio(model, dataloader, max_samples, 
#                                             v_AS*noise_level/torch.norm(v_AS),loss,
#                                             is_small_dataset=is_small_dataset)
#         if attack_ratio >= rho:
#             break
      
#     if itr+1==max_iter and attack_ratio < rho:
#         print('Maximal iteration arrived, current attack ratio {} < {},|v|={}'.format(attack_ratio,rho,torch.norm(v_AS)))
#     return v_AS/(1e-10+torch.norm(v_AS))   


# def AS_attack_input_one_example(model,image_data, image_label_target,
#                          max_num_noise, noise_level, r_max, loss, 
#                         max_iter = 1, attack_target = None,device='cpu'):
#     assert(image_data.shape[0]==1)
#     v_AS = torch.zeros(image_data.shape)
#     image_data_target = image_data.clone()
#     sign_v = 1
#     for itr in range(max_iter):
#         ASlayer, Sigma = get_AS_transform_input_one_example(model, 
#                         image_data + noise_level*v_AS, image_label_target,
#                          max_num_noise, noise_level, r_max, loss, device='cpu')
#         v_AS_i = ASlayer.V[:,0].reshape(v_AS.shape)
#         output_positive = model(image_data+noise_level*(v_AS+v_AS_i))
#         output_negative = model(image_data+noise_level*(v_AS-v_AS_i))
#         if attack_target is None:# to maxmize loss
#             if loss(output_positive, image_label_target)<loss(output_negative, image_label_target):
#                 sign_v = -1
#                 if output_negative.argmax()!=image_label_target:
#                     v_AS += sign_v*v_AS_i
#                     break
#             else:
#                 sign_v = 1
#                 if output_positive.argmax()!=image_label_target:
#                     v_AS += sign_v*v_AS_i
#                     break 
#         else:
#             if loss(output_positive, image_label_target)<loss(output_negative, image_label_target):
#                 sign_v = 1
#                 if output_positive.argmax()==image_label_target:
#                     v_AS += sign_v*v_AS_i
#                     break
#             else:
#                 sign_v = -1
#                 if output_negative.argmax()==image_label_target:
#                     v_AS += sign_v*v_AS_i
#                     break

#         v_AS += sign_v*v_AS_i
        
#     return v_AS/torch.norm(v_AS)

# def compute_attack_ratio_smalldataset(model,dataset,dataset_labels, 
#                      noise,loss,attack_target=None,device='cpu'):
 
    
#     return attack_ratio