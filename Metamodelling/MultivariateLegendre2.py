# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:18:13 2019

@author: babshava
"""

import numpy as np
from scipy.special import comb
import itertools
from pyANOVAMOP.Metamodelling.OrthonormalPolynomialLegendre import orthonormal_polynomial_legendre

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
            for l in range(No2):
                t += 1
                for i in range(j):
                    MultivariateLegendre2[:,t] = MultivariateLegendre2[:,t] * PolynomialEvals[Combinations[k,i]-1][:,int(alpha[l,i])]
            
            
    
    MultivariateLegendre2 = MultivariateLegendre2[:,0:t+1] # check if t+1 needs to be t, instead.
            
    return MultivariateLegendre2 