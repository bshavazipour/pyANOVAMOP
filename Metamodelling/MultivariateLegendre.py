# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:35:30 2019

@author: babshava
"""



"""

"""

import numpy as np
import itertools
from scipy.special import comb


def MultivariateLegendre(D, P, MaxIntOrder):
    """
    Inputs:
              
    Outputs
       """
    (n,d) = np.shape(D)
    M = int(comb(d+P, d))
    storealpha = np.zeros((M-1,d))
    MultivariateLegendre = np.ones((n,M))
    
    if d == 1:
        MultivariateLegendre = orthonormal_polynomial_legendre(P,D) 
        storealpha = np.arange(1,P+1)[np.newaxis].T     
        AnovaIndicators = np.ones((P,1))
        Lambda = np.zeros((2,P))
        Lambda[0,:] = 1
        Lambda[1,:] = storealpha.T
        return (MultivariateLegendre, storealpha, AnovaIndicators, Lambda)
    
    
    if MaxIntOrder < 1 or MaxIntOrder > min(d,P):
        MaxIntOrder = min(d,P)
    else:
        MaxIntOrder = round(MaxIntOrder)
        
    PolynomialEvals = []
    for j in range(d):
        PolynomialEvals.append(orthonormal_polynomial_legendre(P,D[:,j]))
        MultivariateLegendre[:,0] = MultivariateLegendre[:,0] * PolynomialEvals[j][:,0]
    
    Nf = 2**d-1
    for i in range(MaxIntOrder,d):
        Nf = Nf - int(comb(d,i))
        
    AnovaIndicators = np.zeros((Nf,d))
    Lambda = np.zeros((2,M-1))
    r = 0
    t = 0 # as indices start from 0 in python
    u = 0
    
    for j in range(1,MaxIntOrder+1):
        Combinations = np.array(list(itertools.combinations(range(1,d+1), j))) # Compare to combnk() in Matlab it automatically be sorted. 
        No = Combinations.shape[0]  
        alpha2 = np.array(list(itertools.combinations(range(1,P+1), j)))
        No2 = alpha2.shape[0]
        alpha = np.zeros((No2,j))
        alpha[:,0] = alpha2[:,0]
        
        for i in range(1,j):
            alpha[:,i] = alpha2[:,i] - alpha2[:,i-1]
            
        for k in range(No):
            for l in range (No2):
                t += 1
                for i in range(j):
                    MultivariateLegendre[:,t] = MultivariateLegendre[:,t] * PolynomialEvals[Combinations[k,i]-1][:,int(alpha[l,i])]
            
            r += 1
            AnovaIndicators[r-1,Combinations[k,:]-1] = 1
            u2 = u + No2    
            storealpha[u:u2, Combinations[k,:]-1] = alpha
            Lambda[0,u:u2] = r
            u = u2
    
    Lambda[1,:] = [sum(row) for row in storealpha]
    AnovaIndicators = np.array(AnovaIndicators, dtype=bool)
    MultivariateLegendre = MultivariateLegendre[:,0:t+1]
    storealpha = storealpha[0:u,:]
    Lambda = Lambda[:,0:u]
        
        
    return (MultivariateLegendre, storealpha, AnovaIndicators, Lambda)        
        

"""

"""

def MultivariateLegendre2(D, P, MaxIntOrder):
    """
    Inputs:
       
       
    Outputs
    """
    (n,d) = np.shape(D)
    M = int(comb(d+P, d))
    MultivariateLegendre2 = np.ones((n,M))
    
    if d == 1:
        MultivariateLegendre2 = orthonormal_polynomial_legendre(P,D) 
        
        return MultivariateLegendre2
    
    
    if MaxIntOrder < 1 or MaxIntOrder > min(d,P):
        MaxIntOrder = min(d,P)
    else:
        MaxIntOrder = round(MaxIntOrder)
        
    PolynomialEvals = []
    for j in range(d):
        PolynomialEvals.append(orthonormal_polynomial_legendre(P,D[:,j]))
        MultivariateLegendre2[:,0] = MultivariateLegendre2[:,0] * PolynomialEvals[j][:,0]
    
   
    t = 0 # as indices start from 0 in python
       
    for j in range(1,MaxIntOrder+1):
        Combinations = np.array(list(itertools.combinations(range(1,d+1), j))) # Compare to combnk() in Matlab it automatically be sorted. 
        No = Combinations.shape[0]  
        alpha2 = np.array(list(itertools.combinations(range(1,P+1), j)))
        No2 = alpha2.shape[0]
        alpha = np.zeros((No2,j))
        alpha[:,0] = alpha2[:,0]
        
        for i in range(1,j):
            alpha[:,i] = alpha2[:,i] - alpha2[:,i-1]
            
        for k in range(No):
            for l in range (No2):
                t += 1
                for i in range(j):
                    MultivariateLegendre2[:,t] = MultivariateLegendre2[:,t] * PolynomialEvals[Combinations[k,i]-1][:,int(alpha[l,i])]
            
            
    
    MultivariateLegendre2 = MultivariateLegendre2[:,0:t+1]
            
    return MultivariateLegendre2       

"""

"""   
        
        
def orthonormal_polynomial_legendre(p,x):
    """
    Inputs:
        p is P: Max polinomial degree of orthonormal polynomial regressors
        x is a n*1 matrix (a column of the design matrix D e.g. D[:, j])
   
    Output:
        v: Orthonormal polynomial regressors    
    """

    nn = np.arange(1,p+1)[np.newaxis].T
    b = np.append([1], (nn**2/((2*nn-1)*(2*nn+1)))[np.newaxis].T)[np.newaxis].T
    sqrtb = np.sqrt(b)
    
    n = x.shape[0] # Number of rows in x (e.g. if x=np.array([[1,2,3,4]] then, x.shape[0]=1, and x.shape[1]=4)
    
    if p < 0:
        v = []
        return v
          
    v = np.zeros((n,p+1))
    v[:,0]=1/sqrtb[0]
    
    if p < 1:
        return v
    
    v[:,1] = x * v[:,0] / sqrtb[1]
    
    for i in range(1,p):
        v[:, i+1] = (x * v[:, i] - sqrtb[i] * v[:, i-1]) / sqrtb[i+1]
               
    return v    
    
    
    
                
        
        
        
        
        
        
        
        
        
        
        