import numpy as np
#import pdb
#import numpy as np
#from torch.autograd import Variable
import torch as torch
#import copy
from torch.autograd.gradcheck import zero_gradients 


 
def compute_grad_matrix2(x, fx):
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
    #jacobian = jacobian.transpose(0, 1).contiguous()
    # (n_classes x n_samples ) × n_features
    # n_outputs = jacobian.shape[1]
    return jacobian#.view(jacobian.shape[0] * jacobian.shape[1], -1)
 



# The UAP is our baseline from the paper
# S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, 
# Universal adversarial perturbations, in Proceedings of the 
#IEEE conference on computer vision and pattern recognition, 2017, pp. 1765–1773.

def deepfool(image, f,device, grads=compute_grad_matrix2, num_classes=10, overshoot=0.02, max_iter=50): 
    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    
    image = image.to(device)
    f = f.to(device)
 
    with torch.no_grad():
        f_image = f(image)
        
    I = f_image.view(-1).argsort(descending=True) 
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image.clone()

    with torch.no_grad():
        f_i = f(pert_image).view(-1)
    k_i = torch.argmax(f_i)

    w = torch.zeros(input_shape).to(device)
    r_tot = torch.zeros(input_shape).to(device)

    loop_i = 0
    while k_i == label and loop_i < max_iter:
        
        pert = np.inf
        
        pert_image = pert_image.clone().detach().requires_grad_(True)
        res = f(pert_image)
        gradients = grads(pert_image,res)

        for k in range(1, num_classes):

            # set new w_k and new f_k
            w_k = gradients[I[k], :, :, :, :] - gradients[I[0], :, :, :, :]
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/torch.norm(w_k)

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / torch.norm(w)
        r_tot = r_tot+r_i

        # compute new perturbed image
        pert_image = image + (1+overshoot)*r_tot
        loop_i += 1

        # compute new label
        with torch.no_grad():
            f_i = f(pert_image).view(-1)
        k_i = int(torch.argmax(f_i))
    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, k_i, pert_image


def universal_perturbation(dataset, model,est_labels_orig=None, device='cpu',xi=10, delta=0.2, max_iter_uni = 10, 
                           p=2, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    model = model.to(device)
    dataset = dataset.to(device)
    v = 0
    num_images =  dataset.shape[0] # The images should be stacked ALONG FIRST DIMENSION
    dataset_perturbed = dataset.clone()
    if est_labels_orig is None:
        est_labels_orig = torch.zeros((num_images)).to(device)
    else:
        est_labels_orig = est_labels_orig.float().to(device)
        
        
    est_labels_pert = torch.zeros((num_images)).to(device)

    batch_size = 100
    num_batches = num_images // batch_size
    if num_images % batch_size !=0:
        num_batches += 1

        # Compute the estimated labels in batches
    if est_labels_orig is None:   
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            with torch.no_grad():
                est_labels_orig[m:M] = model(dataset[m:M, :, :, :]).argmax(dim=1)

    fooling_rate = 0.0
    
    itr = 0
    
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
#         print('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :,:]
            dr,itrr,_,_ = deepfool(cur_img + v, model, device, 
                                   num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

            # Make sure it converged...
            if itrr < max_iter_df-1:
                vold = v + dr
                v = proj_lp(vold.clone(), xi, p)
                
        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = (dataset + v).to(device)

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            with torch.no_grad():
                est_labels_pert[m:M] = model(dataset_perturbed[m:M, :, :, :]).argmax(dim=1)

        # Compute the fooling rate
        fooling_rate = torch.sum(est_labels_pert != est_labels_orig).double() / num_images
        #print('FOOLING RATE = ', fooling_rate)
        
    print('    UAP: iteration {}/{}, attack ratio {:.2f}/{:.2f}, |v|={:.2f}'.format(
                itr, max_iter_uni,fooling_rate, 1-delta,torch.norm(v)))
    return v   

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/torch.norm(v))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = torch.sign(v) * torch.min(torch.abs(v), torch.ones(v.shape).to(v.device))
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

