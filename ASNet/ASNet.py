import torch
import torch.nn as nn
import copy
import warnings

##  3.3. Active Subspace Network (ASNet)
# including the premodel, the AS layer and the PCE layer

class ASNET(nn.Module):
    def __init__(self, pre_model,AS_model=None, 
                 basis_layer=None,PCE_coeff=None,device='cuda',num_classes=10, AS_model_shape=None):
        super(ASNET, self).__init__()
        self.premodel = copy.deepcopy(pre_model)
        if AS_model is not None:
            self.AS = nn.Linear(AS_model.V.shape[0],AS_model.V.shape[1],  bias=False)
            self.AS.weight.data = copy.deepcopy(AS_model.V).t() 
        elif (AS_model_shape is not None) and (len(AS_model_shape) == 2):
            self.AS = nn.Linear(AS_model_shape[0],AS_model_shape[1],  bias=False)
        else:
            raise ValueError('AS_model and AS_model_shape, At least one of them must be given')
        if basis_layer is None:
            raise ValueError('basis_layer must be given')
        self.PCE_basis =  basis_layer
        self.PCE = nn.Linear(basis_layer.nbasis, num_classes, bias=False) 
        if PCE_coeff is not None:     
            self.PCE.weight.data = copy.deepcopy(PCE_coeff).t()   
    
    def forward(self,x):
        x = self.premodel(x)
        x = x.view(x.size(0),-1)
        x = self.AS(x) 
        x = self.PCE_basis(x)
        x = self.PCE(x)
        
        return x


# To construct the PCE layer as (3.10) in the paper
# We first project the r-dimensional AS output to (r+p,p) basis phi_alpha

class BasisLayer(nn.Module): 
    def __init__(self, PCE_model=None, device=None):
        super(BasisLayer, self).__init__() 
        self.d = PCE_model.d
        self.p = PCE_model.p
        self.nbasis = PCE_model.nbasis
        if device is not None:
            warnings.warn("Device has been deprecated", DeprecationWarning)
        
        if PCE_model is not None:
            self.mean = nn.Parameter(PCE_model.mean, requires_grad=False)
            self.var = nn.Parameter(PCE_model.var, requires_grad=False)
            self.oneDbasis = nn.Parameter(PCE_model.oneDbasis, requires_grad=False )
            self.idxset = nn.Parameter(PCE_model.idxset, requires_grad=False)
        else:
            self.mean = nn.Parameter(0, requires_grad=False)
            self.var = nn.Parameter(0, requires_grad=False)
            self.oneDbasis = nn.Parameter(0, requires_grad=False )
            self.idxset = nn.Parameter(0, requires_grad=False)

    def forward(self,x):
         return self.BasisMat(x)
        
    # Hermite Functions for normal distribution        
    def NormalBasis(self):
        p = self.p
        B = torch.zeros([p+1,p+1])
        B[0,0]=1 # 0nd order
        if p>=1:
            B[1,1] = 2# 1st order
        for i in range(1,p): # i-th order
            B[i+1,1:i+2] = 2*B[i,:i+1]
            B[i+1,:i] -=2*i*B[i-1,:i]
        return B

    
    def PolyVal(self,x,oneDbasis):

        p = self.p
        [n,d] = x.shape
        x_pows = torch.zeros((n,d,p+1),dtype=torch.float32)
        for i in range(p+1):
            x_pows[:,:,i] = x**i

        polyval = torch.zeros((n,d,p+1),dtype=torch.float32)
        for ip in range(p+1):
            for i in range(ip+1):
                if oneDbasis[ip,i]!=0:
                    polyval[:,:,ip] += oneDbasis[ip,i]*x_pows[:,:,i]

        return polyval.to(x.device)
    
    
    def BasisMat(self,x):
        mean = self.mean
        var = self.var
        idxset = self.idxset
        p = self.p
        d = self.d
        oneDbasis = self.oneDbasis
        assert(len(mean)==d)
        assert(len(var)==d)
        for i in range(d):
            x[:,i] = (x[:,i]-mean[i])/var[i]

        oneDpolyval = self.PolyVal(x,oneDbasis)
       
        Phi = torch.ones([x.shape[0], self.nbasis], 
                         dtype=torch.float32).to(x.device)
        for j in range(d):
            Phi *= oneDpolyval[:,j,idxset[:,j]]

        return Phi



    
