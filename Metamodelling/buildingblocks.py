# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:50:48 2019

@author: babshava
"""
import numpy as np

def buildingblocks(X0,Coefficients,Choice):
    """
    Coefficients ?
    Choice ?
    """
    [n, d] = X0.shape
    X = (X0 +1 ) / 2

    L = Choice.shape[0]
    buildingblocks = 0
    
    for i in range(L):
        Int = np.where(Choice[i,2:d+2])[0]
        if Choice[i,1] == 1:
            Xint = (X[:,Int],2).prod(0)
        else:
            Xint = (X[:,Int]).mean(1)
        
        buildingblocks += Coefficients[i] * subfunction(Xint,Choice[i,0])  # check the product
    
    

def subfunction(x,no):
    """
    
    """
        
    if no==1:
        subfunction = x          
    elif no==2:
        subfunction = (2 * x - 1) ** 2   #check the product
    elif no==3:
        subfunction = np.sin(2 * np.pi * x) / (2 - np.sin(2 * np.pi *x))       
    elif no==4:
        subfunction = 0.1 * np.sin(2 * np.pi * x) + 0.2 * np.cos(2 * np.pi * x) + 0.3 * np.sin(2 * np.pi * x) ** 2 + 0.4 * np.cos(2 * np.pi * x) ** 3 + 0.5 * np.sin(2 * np.pi * x) ** 3        
    
    return subfunction