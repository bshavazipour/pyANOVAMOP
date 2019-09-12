# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:52:10 2019

@author: babshava
"""
from pyANOVAMOP.Metamodelling.SurrogatePrediction import SurrogatePrediction
import numpy as np


def ASF(x0, #Input  initial guess
        #model  # model = SurrogateDataInfo has all info about the all objectives returend from the BPC; SurrogateDataInfo[i] has the info of the i-th objectives, e.g. SurrogateDataInfo[i].md  
        #DataSets,#[objective][0] 
        #Y[objective] 
        P,#[objective]
        md,#[objective] 
        check3,#[objective] 
        MaxIntOrder,#[objective] 
        #iteration[objective] 
        ObjIndices, #SubProblemObjectiveIndices,
        #DecomposedBounds,  #VariableBounds
        z
):
    """
    Achievement Scalarization function
        
    """
    #xtemp = x#.T 
    #NumObj = len(ObjIndices)
      
    #numPop = xtemp.shape[0] #DataSets[0][0].shape[0]
    #frange = DecomposedBounds[1,:] - DecomposedBounds[0,:] # ub - lb
    #xtemp = ((xtemp+1) * np.matlib.repmat(frange,numPop,1)) / 2 + np.matlib.repmat(DecomposedBounds[0,:],numPop,1)
    
    #y = [] --> y.append() 
    y = np.zeros(len(ObjIndices))
    i = 0
    
    for objective in ObjIndices: # range(NumObj) 
        y[i] = float(SurrogatePrediction(np.matrix(x0), # must be a matrix
                                         #SurrogateDataInfo[objective][0]
                                         #DataSets[objective][0] 
                                         #Y[objective] 
                                         P[objective],
                                         md[objective], 
                                         check3[objective], 
                                         MaxIntOrder[objective],
                                         ) 
        )
        i += 1
        


    ASFval = (y-z).max(0) + (10 ** (-6)) * (y-z).sum() # axis=1


    return ASFval









"""


def SubProblem_objFun(x0,SubProblemObjectiveIndices,SubProblemVariablesIndices,FixedIndices,FixedValues,VariableBounds,model):
    
    #x=x0
    
    
    numPop = x0.shape[0]
    frange = VariableBounds[1,:] - VariableBounds[0,:] # ub - lb
    np.delete(frange, frange[:,FixedIndices]) # remove a column FixedIndices
    x0 = ((x0+1) * np.matlib.repmat(frange,numPop,1)) / 2 + np.matlib.repmat(VariableBounds[0,SubProblemVariablesIndices],numPop,1)
    numVar = len(SubProblemVariablesIndices) #+len(FixedIndices)
    x = np.zeros((numPop,numVar))
    x[:,SubProblemVariablesIndices] = x0
    x[:,FixedIndices] = FixedValues
    numObj = len(SubProblemObjectiveIndices)
    y = np.zeros(1,numObj)
    cons = [] # check if it should not be a list
    
    for objective in range(numObj):
        y[objective] = SurrogatePrediction(x,model(SubProblemObjectiveIndices[objective]))

    return (y, cons)
    
"""