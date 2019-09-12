# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:50:02 2019

@author: babshava
"""

import numpy as np
from pyANOVAMOP.Metamodelling.MultivariateLegendre2 import MultivariateLegendre2 #, MultivariateLegendre # ,orthonormal_polynomial_legendre,


def SurrogatePrediction(x00, 
                        #model: # model stored as Data includes e.g. md, check3, P, MaxIntOrder
                        #DataSets[objective][0], 
                        #Y[objective] 
                        P,#[objective],
                        md,#[objective], 
                        check3,#[objective], 
                        MaxIntOrder #[objective], 
                        #iteration[objective]
):
    """
    Estimate the objective functions for the given solution x0
    """
    #md = model.md
    #check3 = model.check3
    #P = model.P
    #MaxIntOrder = model.MaxIntOrder
    x0 = MultivariateLegendre2(x00,P,MaxIntOrder)
    x = x0[:,check3]
    Pred = np.matmul(x,md) # check if the right product has been used here
    
    return Pred

