# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:04:47 2019

@author: babshava
"""
import numpy as np



class Problem_Bab(BaseProblem):
    def __init__(self, 
            
            
    )
    
    

def SubProblem(SubProblemObjectiveIndices,SubProblemVariablesIndices,Bounds,lb,ub,FixedIndices,FixedValues,model):
    """
    description
    Solving a sub problem by calling the solver
    
    SubProblemObjectiveIndices denotes the number of rows for the active objectives in this sub-problem (from the original problem)
    SubProblemVariablesIndices denotes the number of columns for the active variables in this sub-problem (from the original problem)
    Sub problem_lb = Bounds[0,:]
    Sub problem_ub = Bounds[1,:]
    lb ub are the lower and upper bounds of the original problem
    FixedIndices,
    FixedValues,
    model
    """
    NumObj = len(SubProblemObjectiveIndices)
    NumVar = len(SubProblemVariablesIndices)
    #Calling the solver  Main is RVEA
    #[x, f] = Main('Surrogate', SubProblemObjectiveIndices,SubProblemVariablesIndices, NumObj, NumVar, Bounds, lb, ub, FixedIndices, FixedValues, model)
    [x, f] = Problem_Bob('Surrogate', SubProblemObjectiveIndices,SubProblemVariablesIndices, NumObj, NumVar, Bounds, lb, ub, FixedIndices, FixedValues, model)
    
    return (x, f)










def SubProblem_objFun(x0,SubProblemObjectiveIndices,SubProblemVariablesIndices,FixedIndices,FixedValues,VariableBounds,model):
    """
    x=x0
    """
    
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