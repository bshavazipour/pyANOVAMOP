# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:50:02 2019

@author: babshava
"""

import numpy as np

def SurrogatePrediction(x0, model): # model stored as Data includes e.g. md, check3, P, MaxIntOrder
    """
    Evaluate the objective functions for the given solution x0
    """
    md = model.md
    check3 = model.check3
    P = model.P
    MaxIntOrder = model.MaxIntOrder
    x0 = MultivariateLegendre2(x0,P,MaxIntOrder)
    x0 = x0[:,check3]
    Pred = np.matmul(x0,md) # check if the right product has been used here
    
    return Pred

