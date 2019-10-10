import torch
import pdb
from torch import nn
import scipy.misc
from sklearn.linear_model import LinearRegression
#from copy import deepcopy
import numpy as np

class PCEModel():
    def __init__(self,mean,var,d=50,p=2,device=None):
        #super(PCEModel, self).__init__()
        self.d = d
        self.p = p
        self.mean = mean
        self.var = var
        self.nbasis = scipy.special.comb(d+p,p).astype(int)
        if device==None:
            self.device = 'cpu'
        else:
            self.device = device
        self.oneDbasis = self.NormalBasis() 
        self.idxset = indexset(d,0)
        for i in range(1,p+1):
            self.idxset = torch.cat((self.idxset,indexset(d,i)),dim=0)
    
    # Basis Functions for normal distribution        
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

        return polyval.to(self.device)
    
    
    def BasisMat(self,x):
        mean = self.mean
        var = self.var
        idxset = self.idxset
        p = self.p
        d = len(mean)
        oneDbasis = self.oneDbasis
        assert(len(var)==d)
        for i in range(d):
            x[:,i] = (x[:,i]-mean[i])/var[i]

        oneDpolyval = self.PolyVal(x,oneDbasis)
       
        Phi = torch.ones([x.shape[0], self.nbasis], dtype=torch.float32).to(self.device)
        for j in range(d):
            Phi *= oneDpolyval[:,j,idxset[:,j]]
        return Phi

    def Training(self, x,y,label): 
        Phi = self.BasisMat(x)
        
        if type(Phi)!=np.ndarray:
            Phi = Phi.cpu().detach().numpy()
        if type(y)!=np.ndarray:
            y = y.cpu().detach().numpy()
        if type(label)!=np.ndarray:
            label = label.cpu().numpy()
            
        LR = LinearRegression(fit_intercept=False).fit(Phi,y)

        score_approx = LR.score(Phi,y)
        coeff = LR.coef_.transpose() 
        y_PCE = Phi@coeff
        score_label = (label == np.argmax(y_PCE,axis=1)).mean()
        return coeff,  score_approx, score_label
    
    def Inference(self,x,coeff):
        Phi = self.BasisMat(x)
        if Phi.shape[1]==coeff.shape[0]:
            y = Phi@coeff.to(self.device)
        else:
            y = Phi@coeff.t().to(self.device)
        return y
            

def indexset(d,p):
    if d==1:
        IdxMat = p*torch.ones((1,1), dtype=torch.int64)
    else:
        for i in range(p+1):
            Idx_tmp = indexset(d-1,p-i) 
            sz = Idx_tmp.shape[0]
            Idx_tmp = torch.cat((i*torch.ones((sz,1), dtype=torch.int64),Idx_tmp),dim=1)              
            if i==0:
                IdxMat = Idx_tmp
            else:
                IdxMat = torch.cat((IdxMat,Idx_tmp),dim=0)

    return IdxMat  

# def PCE(X_train, Y_train, porder):
#     if type(X_train)!=np.ndarray:
#         X_train = X_train.numpy()
#     if type(Y_train)!=np.ndarray:
#         Y_train = Y_train.numpy()
#     meanX = np.mean(X_train,0)
#     stdX = np.diag(np.std(X_train,0))
#     distribution = cp.MvNormal(meanX,stdX)
#     polynomial_expansion = cp.orth_ttr(porder, distribution)
#     foo_approx = cp.fit_regression(polynomial_expansion, np.transpose(X_train), Y_train)
#     return foo_approx

# def PCE_approx(X, foo_approx):
#     if type(X)!=np.ndarray:
#         X = X.numpy()
#     Y_PCE = foo_approx(*np.transpose(X)).T
#     Label_PCE = np.argmax(Y_PCE,axis=1)
#     return Y_PCE,Label_PCE


# ASNet
 

# class ASNET(nn.Module):
#     def __init__(self, premodel,num_neurons, p,mean,var,nAS, num_basis, num_classes=10):
#         super(ASNET, self).__init__()
#         self.premodel = premodel
#         self.AS = nn.Linear(nAS, num_neurons, bias=False)
#         self.PCE = nn.Sequential(BasisLayer(p,mean,var),
#                                  nn.Linear(num_basis,num_classes, bias=False))
    
#     def forward(self,x):
#         x = self.premodel(x)
#         x = x.view(x.size(0),-1)  
#         x = self.AS(x)  
#         x = self.PCE(x)
        
#         return x


# class BasisLayer(nn.Module): 
#     def __init__(self,p,mean,var):
#         super(BasisLayer, self).__init__()
#         self.p = p
#         self.mean = mean
#         self.var = var 
#         self.HPCE = HermitePCE(len(self.mean),self.p)
#     def forward(self,x):
#          return self.HPCE.BasisMat(x,self.mean,self.var)
    
# class PCELayer(nn.Module):
#     def __init__(self,PCECoeff,device='cuda'):
#         super(PCELayer, self).__init__() 
#         self.fc = nn.Sequential(
#              BasisLayer(maxpower),
#              nn.Linear(nbasis, num_classes))
        
# #         self.coeff = PCECoeff
# #         self.PCECoeff = nn.Parameter(self.coeff.to(device))
# #         self.PCEBasis = HermitePCE(len(mean),p)
        
#     def forward(self,x):
#         x = x.view(x.shape[0], -1)#.to(self.device)
#         Phi = self.PCEBasis.BasisMat(x,self.mean,self.var)
#         y = Phi@self.PCECoeff
#         return y

#     def backward(self,y):

# class PCELayer(nn.Module):
#     def __init__(self,p,mean,var,PCECoeff,device='cuda'):
#         super(PCELayer, self).__init__()
#         self.p = p
#         self.mean = mean
#         self.var = var
#         self.nbasis = len(mean)*(len(mean)+1)/2
#         self.device = device
#         self.coeff = PCECoeff
#         self.HPCE = HermitePCE(len(mean),p)
        

#     def forward(self,x):
#         x = x.view(x.shape[0], -1)#.to(self.device)
#         Phi = self.HPCE.BasisMat(x,self.mean,self.var)
#         #y = Phi@self.PCECoeff
#         y = self.linearreg(Phi)
#         return y
   #     def GBasis(self):
#         Basis = self.oneDbasis
#         p = self.p
#         GB = torch.rand_like(Basis)
#         for i in range(p):
#             GB[:,i] = Basis[:,i+1]*(i+1)
#         GB[:,-1]=0
#         return GB

#     def GBasisMat(self,x):
#         idxset = self.idxset
#         oneDbasis = self.oneDbasis
#         GoneDbasis = self.GoneDbasis
#         mean = self.mean
#         var = self.var
#         p = self.p
#         d = self.d
#         assert(len(mean)==d)
#         assert(len(var)==d) 
#         for i in range(d):
#             x[:,i] = (x[:,i]-mean[i])/var[i]

#         oneDpolyval = self.PolyVal(x,oneDbasis)
#         GoneDpolyval = self.PolyVal(x, GoneDbasis)
#         Phi = torch.ones([x.shape[0],self.nbasis,d]).to(self.device)

#         for j in range(d):
#             for j2 in range(j):
#                 Phi[:,:,j] *= oneDpolyval[:,j2,idxset[:,j2]]
#             Phi[:,:,j] *= GoneDpolyval[:,j,idxset[:,j]]
#             for j2 in range(j+1,d):
#                 Phi[:,:,j] *= oneDpolyval[:,j2,idxset[:,j2]]
#         return Phi
        
        # x=student.premodel(data)
        # x = x.view(x.size(0),-1)
        # z= student.AS(x)
        # Gz = student.PCE_model.GBasisMat(z)
        # cc = student.PCE.weight
        # Gv = student.AS.weight.grad  
        # tt = torch.mm(cc,Gz[0,:])
        # myGv = torch.ger(x[0,:],tt[0,:])
        # torch.norm(Gv-myGv.t())/torch.norm(Gv)

# index set for p-th order multi-variate

 