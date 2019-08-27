# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:38:41 2019

@author: babshava

Myfun to evaluate the objective values
"""

from pyANOVAMOP.CommonFiles.P_objective import P_objective

import numpy as np
#import numpy.matlib

def MyFun(X0,lb,ub,ProbName,ObjInd,ObjNum): 
    """
    evaluate the objective values
    return the values for the objective which has index of ObjInd
    
    X0 is the dataset D
    """

    frange = ub - lb
    n = X0.shape[0]
    X = ((X0+1) * np.matlib.repmat(frange,n,1)) / 2 + np.matlib.repmat(lb,n,1)
    fTemp = P_objective('value',ProbName,ObjNum,X)
    Value = fTemp[:,ObjInd] 
    
    return Value
    