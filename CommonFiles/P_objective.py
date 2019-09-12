# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:51:38 2019

@author: babshava
"""

"""
Test Problems  ----  P_objective

"""
import numpy as np
#import math

#def P_objective(
#        Operation,
#        Problem,
#        M,
#        Input
#        ):#,
        #SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model
        #):
#    """
        
#    """
    
    #k = np.nonzero(~isstrprop(Problem,'digit'),1,'last');
    #switch Problem(1:k)
    #if Problem == 'ZDT':
    #    [Output,Boundary,Coding]=P_ZDT(Operation,Problem,Input)
    #elif Problem == 'DTLZ':
    #    [Output,Boundary,Coding] = P_DTLZ(Operation,Problem,M,Input)
    #elif Problem == 'SDTLZ':
    #    [Output,Boundary,Coding] = P_SDTLZ(Operation,Problem,M,Input)
#    if Problem == 'ALBERTO':
#        Output = P_ALBERTO(Operation,Input) # [Output,Boundary,Coding]
    #elif Problem == 'Surrogate':
    #    [Output,Boundary,Coding] = P_Surrogate(Operation,M,Input,SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model)            
    #else:
    #    raise Exception(Problem,'Not Exist'.format(x))
        
     
#    return Output#,Boundary,Coding)
    


def P_objective(Input,k):

    """    
    An example in section 4.1 of the ANOVA-MOP paper (p 3280)
    
    We consider the following ﬁve-objective pa-
    rametrized optimization problem 5 × 5 ≃ (3 × 3) ⊗ (2 × 2) with λ = 1.3:

    (17) minimize  {f1(x), . . . , f5(x)}, where x = (x1, . . . , x5) and
         x∈[−λ,λ]


    f1(x) = g1(x) + γg4(x),              g1(x) = 
(x1, x2, x3)T − P1T 
2 , P1 = (1, 1, 1)T ,
    f2(x) = g2(x) + γg5(x),              g2(x) = 
(x1, x2, x3)T − P2T 
2 , P2 = (1, −1, −1)T ,
    f3(x) = g3(x) + γ(g4(x) + g5(x)),    g3(x) = 
(x1, x2, x3)T − P3T 
2 , P3 = (1, 1, −1)T ,
    f4(x) = g4(x) + γg1(x),              g4(x) = 
(x4, x5)T − P4T 
2 , P4 = (1, −1)T ,
    f5(x) = g5(x) + γ(g1(x) + g2(x)),    g5(x) = 
(x4, x5)T − P5T 
2 , P5 = (−1, 1)T ,
    with γ = 7.000×10−3.
    """

    #Boundary = math.nan
    #Coding = math.nan
    
    #Population Initialization
    #if Operation == 'init':
        
    #    numVar = 5
    #    Min = -np.ones((1,numVar))
     #   Max = np.ones((1,numVar))
    #    Population = np.random.rand(Input,numVar) # Input must be a number # create a random matrix
        #Population=((Input+1).*repmat(range,Input,1))/2+repmat(Min,Input,1);
    #    Population = Population * np.matlib.repmat(Max,Input,1) + (1 - Population) * np.matlib.repmat(Min,Input,1)
    #    Output = Population
    #    Boundary = np.array([[Min],[Max]])
    #    Coding = 'Real'

    #if Operation == 'value':
    
    numSample, numVar = np.shape(Input)
    x = Input
        
    epsilon = 0.007 #0.1
        
    P1 = np.array([1, 1, 1])
    P2 = np.array([1, -1, -1])
    P3 = np.array([1, 1, -1])
    P4 = np.array([1, -1])
    P5 = np.array([-1, 1])
        
    Phi1 = ((x[:,0:3] - np.ones((numSample,1)) * P1) ** 2).sum(axis=1)
    Phi2 = ((x[:,0:3] - np.ones((numSample,1)) * P2) ** 2).sum(axis=1)
    Phi3 = ((x[:,0:3] - np.ones((numSample,1)) * P3) ** 2).sum(axis=1)
    Phi4 = ((x[:,3:5] - np.ones((numSample,1)) * P4) ** 2).sum(axis=1)
    Phi5 = ((x[:,3:5] - np.ones((numSample,1)) * P5) ** 2).sum(axis=1)
        
    Output = np.empty((numSample,k))
    Output[:,0] = Phi1 + epsilon * Phi4
    Output[:,1] = Phi2 + epsilon * Phi5
    Output[:,2] = Phi3 + epsilon * (Phi4 + Phi5)
    Output[:,3] = Phi4 + epsilon * Phi1
    Output[:,4] = Phi5 + epsilon * (Phi1 + Phi2)

    return Output #,Boundary,Coding)


def P_objective2(Input,k):

    """    
    An example in section 4.2 of the ANOVA-MOP paper (p 3283)
    
    Next we consider a pa-
    rametrized 12 × 10 ≃ (3 × 3) ⊗ (3 × 3) ⊗ (2 × 2) ⊗ (4 × 2) problem involving twelve
    variables with S = [−λ, λ]12, where λ = 1.3. We minimize ten objectives of component
    functions deﬁned as follows with γ = 7.000 × 10−3:
        
        f1(x) = g1(x) + γg4(x), g1(x) = 
(x1, x2, x3)T − P1T 
2 , P1 = (1, 1, 1)T ,
        f2(x) = g2(x) + γg5(x), g2(x) = 
(x1, x2, x3)T − P2T 
2 , P2 = (1, −1, −1)T ,
        f3(x) = g3(x) + γ(g4(x) + g6(x)), g3(x) = 
(x1, x2, x3)T − P3T 
2 ,

                          P3 = (1, 1, −1)T ,
        f4(x) = g4(x) + γ(g1(x) + g7(x)), g4(x) = 
(x4, x5, x6)T − P4T 
2 ,

                        P4 = (−1, −1, −1)T ,
        f5(x) = g5(x) + γ(g2(x) + g8(x)), g5(x) = 
(x4, x5, x6)T − P5T 
2 ,

                 P5 = (−1, 1, −1)T ,
        f6(x) = g6(x) + γ(g3(x) + g9(x)), g6(x) = 
(x4, x5, x6)T − P6T 
2 , P6 = (−1, −1, 1)T ,
        f7(x) = g7(x) + γ(g2(x) + g5(x)), g7(x) = 
(x7, x8)T − P7T 
2 , P7 = (1, −1)T ,
        f8(x) = g8(x) + γ(g1(x) + g4(x) + g9(x)), g8(x) = 
(x7, x8)T − P8T 
2 , P8 = (−1, 1)T ,

        f9(x) = g9(x) + γ(g3(x) + g6(x) + g8(x)), g9(x) = Sum( sin(xi) + cos(xi) ) , xi = 9, ...,12

        f10(x) = g10(x) + γ(g4(x) + g5(x)), g10(x) = Sum( sin(-xi) + cos(-xi) ) , xi = 9, ..., 12
  
    """

    
    numSample, numVar = np.shape(Input)
    x = Input
        
    epsilon = 0.007 #0.1  = γ
        
    P1 = np.array([1, 1, 1])
    P2 = np.array([1, -1, -1])
    P3 = np.array([1, 1, -1])
    P4 = np.array([-1, -1, -1])
    P5 = np.array([-1, 1, -1])
    P6 = np.array([-1, -1, 1])
    P7 = np.array([1, -1])
    P8 = np.array([-1, 1])
        
    Phi1 = ((x[:,0:3] - np.ones((numSample,1)) * P1) ** 2).sum(axis=1)  # g1
    Phi2 = ((x[:,0:3] - np.ones((numSample,1)) * P2) ** 2).sum(axis=1)  # g2
    Phi3 = ((x[:,0:3] - np.ones((numSample,1)) * P3) ** 2).sum(axis=1)
    Phi4 = ((x[:,3:6] - np.ones((numSample,1)) * P4) ** 2).sum(axis=1)
    Phi5 = ((x[:,3:6] - np.ones((numSample,1)) * P5) ** 2).sum(axis=1)
    Phi6 = ((x[:,3:6] - np.ones((numSample,1)) * P6) ** 2).sum(axis=1)
    Phi7 = ((x[:,6:8] - np.ones((numSample,1)) * P7) ** 2).sum(axis=1)
    Phi8 = ((x[:,6:8] - np.ones((numSample,1)) * P8) ** 2).sum(axis=1)
    Phi9 = np.sin(x[:,8:13]).sum() + np.cos(x[:,8:13]).sum() # check if there is need to change x_i values to radian/degrees
    Phi10 = np.sin(-x[:,8:13]).sum() + np.cos(-x[:,8:13]).sum()

    
    Output = np.empty((numSample,k))
    Output[:,0] = Phi1 + epsilon * Phi4
    Output[:,1] = Phi2 + epsilon * Phi5
    Output[:,2] = Phi3 + epsilon * (Phi4 + Phi6)
    Output[:,3] = Phi4 + epsilon * (Phi1 + Phi7)
    Output[:,4] = Phi5 + epsilon * (Phi2 + Phi8)
    Output[:,5] = Phi6 + epsilon * (Phi3 + Phi9)
    Output[:,6] = Phi7 + epsilon * (Phi2 + Phi5)
    Output[:,7] = Phi8 + epsilon * (Phi1 + Phi4 + Phi9)
    Output[:,8] = Phi9 + epsilon * (Phi3 + Phi6 + Phi8)
    Output[:,9] = Phi10 + epsilon * (Phi4 + Phi5)
    
    
    
    return Output #,Boundary,Coding)





