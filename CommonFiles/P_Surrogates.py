# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:50:12 2019

@author: babshava
"""

from pyANOVAMOP.Metamodelling import MultivariateLegendre2 #, MultivariateLegendre # ,orthonormal_polynomial_legendre,


class P_Surrogates(BaseProblem):
    """
    
    """
    
    def __init__(
        self,
        
            
    ):
    
        
        
    def SurrogatePrediction(x00, 
                        #model: # model stored as Data includes e.g. md, check3, P, MaxIntOrder
                        #DataSets[objective][0], 
                        #Y[objective] 
                        P,#[objective],
                        md,#[objective], 
                        check3,#[objective], 
                        MaxIntOrder,#[objective], 
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
    
    

    def objectives(self, decision_variables):
        """Objectives function to use in optimization.
        Parameters
        ----------
        decision_variables : ndarray
            The decision variables
        Returns
        -------
        objectives : ndarray
            The objective values
        """
        objectives = []
        for obj in self.y:
            objectives.append(
                self.models[obj][0].predict(decision_variables.reshape(1, -1))[0]
            )

        return objectives