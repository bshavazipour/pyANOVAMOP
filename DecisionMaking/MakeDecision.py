# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:50:36 2019

@author: babshava
"""


from scipy.optimize import minimize
import numpy as np
import random

from pyANOVAMOP.DecisionMaking.ASF2 import ASF
from pyANOVAMOP.Metamodelling.SurrogatePrediction import SurrogatePrediction



def MakeDecision(z,
                 P,#[objective]
                 md,#[objective] 
                 check3,#[objective] 
                 MaxIntOrder,#[objective] 
                 ObjIndices, #SubProblemObjectiveIndices,
                 Input
):
    """
     
    # Initialization for DIRECT method
    opts.es = 1e-4
    opts.maxevals = 500
    opts.maxits = 500
    opts.maxdeep = 1000
    opts.testflag = 0
    opts.showits = 0
    
    
        
     
    #call an optimizer e.g. minimize or fsolve
    
    # ASF will be called within the optimizer
    #Problem.f ='ASF' # objective function from ASF
   
    
    # Solve the single objective optimizer to Min ASF(X,Z) of the (sub-)problem and get the solution
    #[~, xOptApp] = Direct(Problem,VariableBounds',opts,StrucData)
    """
    
    xOptApp = minimize(ASF, 
                       random.choice(Input), # Only send one sample from the dataset
                       args=(#model  # model = SurrogateDataInfo has all info about the all objectives returend from the BPC; SurrogateDataInfo[i] has the info of the i-th objectives, e.g. SurrogateDataInfo[i].md  
                             #DataSets[0][0][0], 
                             #Y[objective] 
                             P,#[objective]
                             md,#[objective] 
                             check3,#[objective] 
                             MaxIntOrder,#[objective] 
                             #iteration[objective] 
                             ObjIndices, #SubProblemObjectiveIndices,
                             #DecomposedBounds,  #VariableBounds
                             z
                             )
                      )
                         
    #xOptApp = xOptApp.T
    
    
    # Function evluations with new solution
    fOptApp = np.zeros(len(ObjIndices))
    #fOptApp = []
    i = 0
    for objective in ObjIndices: #len(ObjIndices) 
        fOptApp[i] = float(SurrogatePrediction(np.matrix(xOptApp.x),
                                               #SurrogateDataInfo[SubProblemObjectiveIndices[Objective]]
                                               #model: # model stored as Data includes e.g. md, check3, P, MaxIntOrder
                                               #DataSets[objective][0], 
                                               #Y[objective] 
                                               P[objective],
                                               md[objective], 
                                               check3[objective], 
                                               MaxIntOrder[objective] 
                                               #iteration[objective]
                                               )
                           )
        i += 1
        
    return (xOptApp.x, fOptApp)