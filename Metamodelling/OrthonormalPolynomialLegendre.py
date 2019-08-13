# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:36:19 2019

@author: babshava
"""
import numpy as np

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
    
    
    
        