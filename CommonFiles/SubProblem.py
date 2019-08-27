# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:04:47 2019

@author: babshava
"""
import numpy as np


"""
class Problem_Bab(BaseProblem):
    """
    Problem description
    """
    def __init__(self, 
            
            
    )
    
    DataSets[objective][0] 
    #Y[objective] 
    P[objective]
    md[objective] 
    check3[objective] 
    MaxIntOrder[objective] 
    #iteration[objective] 
    #
"""

def SubProblem(SubProblemObjectiveIndices,
               SubProblemVariablesIndices,
               Bounds,
               lb,
               ub,
               FixedIndices,
               FixedValues,
               #model  # model = SurrogateDataInfo has all info about the all objectives returend from the BPC; SurrogateDataInfo[i] has the info of the i-th objectives, e.g. SurrogateDataInfo[i].md  
               DataSets,#[objective][0] 
               #Y[objective] 
               P, #[objective]
               md, #[objective] 
               check3, #[objective] 
               MaxIntOrder #[objective] 
               #iteration[objective] 
               ): 
    """
    
    Building a sub problem
    Then, send it to the solver and get the solutions as well as related objective values
    
    Generating the initial samples for this sub-problem and get the estimated values for active objectives by utilizing relevant surrogets which found via BPC
    
    Args:
        SubProblemObjectiveIndices denotes the number of rows for the active objectives in this sub-problem (from the original problem); e.g. [0,2,4]
        SubProblemVariablesIndices denotes the number of columns for the active variables in this sub-problem (from the original problem); e.g. [0,1,2]
        Sub problem_lb = Bounds[0,:]
        Sub problem_ub = Bounds[1,:]
        lb ub are the lower and upper bounds of the original problem
        FixedIndices,
        FixedValues,
        model
    """
    NumObj = len(SubProblemObjectiveIndices) # e.g. 3
    NumVar = len(SubProblemVariablesIndices) # e.g. 3
    
    # Building sub-problem  (RVEA does not need this initial input)
    NumPop = DataSets[0][0].shape[0]
    InputTemp = np.zeros((NumPop,NumVar + len(FixedIndices)))
    InputTemp[:,FixedIndices] = np.matlib.repmat(FixedValues,NumPop,1)
    InputTemp[:,SubProblemVariablesIndices.astype(int)] = DataSets[0][0][:,SubProblemVariablesIndices.astype(int)]
    
    # New sample (X) for the sub problem
    Input = MapSamples(InputTemp, np.vstack((-np.ones((1,len(lb[0]))), np.ones((1,len(lb[0]))))), np.vstack((lb,ub)))  
    
    SubInput = np.delete(Input, FixedIndices,1) # remove non-active variables (columns in FixedIndices)

    # evaluate the samples and get the estimated objective values from the surrogates   
    Output = [] #np.zeros((NumPop,lb.shape[1])) # lb.shape[1]: number of objectives in the original problem
    
    for objective in SubProblemObjectiveIndices:#range(len(SubProblemObjectiveIndices)):
        Output.append(SurrogatePrediction(Input, # Input also can be used
                                          #SurrogateDataInfo[objective][0]
                                          #DataSets[objective][0] 
                                          #Y[objective] 
                                          P[objective],
                                          md[objective], 
                                          check3[objective], 
                                          MaxIntOrder[objective] 
                                          #iteration[objective]
                                          ) 
                      )

    
    return (Input, SubInput, Output)
    
    
    
    """
    #Calling the solver  Main is RVEA
    #[x, f] = Main('Surrogate', SubProblemObjectiveIndices,SubProblemVariablesIndices, NumObj, NumVar, Bounds, lb, ub, FixedIndices, FixedValues, model)
    [x, f] = P_Surrogate(NumObj, 
                         NumVar,
                         'RVEA', 
                         SubProblemObjectiveIndices,
                         SubProblemVariablesIndices, 
                         Bounds[0,:], 
                         Bounds[1,:], 
                         #model
                         )
    
    return (x, f)
    """


def MapSamples(Samples,NewBounds,CurrentBounds):
    
    """
    
    """
    
    NumSamples = Samples.shape[0]
    LowerBoundNewRange = NewBounds[0,:]
    UpperBoundNewRange = NewBounds[1,:]
    LowerBoundOldRange = CurrentBounds[0,:]
    UpperBoundOldRange = CurrentBounds[1,:]
    m = (UpperBoundNewRange - LowerBoundNewRange) / (UpperBoundOldRange - LowerBoundOldRange)
    Output = np.matlib.repmat(m,NumSamples,1) * (Samples - np.matlib.repmat(LowerBoundOldRange,NumSamples,1)) + np.matlib.repmat(LowerBoundNewRange,NumSamples,1)

    return Output





"""

def SubProblem_objFun(x0,SubProblemObjectiveIndices,SubProblemVariablesIndices,FixedIndices,FixedValues,VariableBounds,model):
    """
    #x=x0
    """
    
    numPop = x0.shape[0]
    frange = VariableBounds[1,:] - VariableBounds[0,:] # ub - lb
    np.delete(frange, FixedIndices,0) # remove a column FixedIndices
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