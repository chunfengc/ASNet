import torch
#import pdb
#from torch import nn
import scipy.misc
from sklearn.linear_model import LinearRegression
#from copy import deepcopy
import numpy as np

# The PCE layer as given in step 2 of Algorithm 3.1 
# The training procedure of the active subspace network (ASNet)
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
        #p = self.p
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

