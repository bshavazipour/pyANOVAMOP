# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:13:44 2019

@author: babshava
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:11:23 2019

@author: babshava
"""

from pyrvea.Population.Population import Population
from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.NSGAIII import NSGAIII

from pyANOVAMOP.Metamodelling.SurrogatePrediction import SurrogatePrediction

import numpy as np

class ANOVAMOPtest1SubpSurrogate(baseProblem):
    """
      New problem description.
    """
    #global SubProblemObjectiveIndices, SubProblemVariablesIndices, P, md, check3, MaxIntOrder 
    
    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
        SubProblemObjectiveIndices = [],
        SubProblemVariablesIndices = [],
        P = [],
        md = [], 
        check3 = [], 
        MaxIntOrder = [],
    ):    
    
        self.SubProblemObjectiveIndices,
        self.SubProblemVariablesIndices,
        self.P, 
        self.md, 
        self.check3, 
        self.MaxIntOrder
        super().__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,  
        )
    
    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions.
        Args:
            decision_variables: a sample  
        """
        
        x = decision_variables
        
        #NumObj = len(SubProblemObjectiveIndices) # e.g. 3
        #NumVar = len(SubProblemVariablesIndices) # e.g. 3
               
        y = np.zeros(len(self.SubProblemObjectiveIndices))
        i = 0
    
        for objective in self.SubProblemObjectiveIndices: # range(NumObj) 
            y[i] = float(SurrogatePrediction(np.matrix(x), 
                                         #SurrogateDataInfo[objective][0]
                                         #DataSets[objective][0] 
                                         #Y[objective] 
                                         self.P[objective],
                                         self.md[objective], 
                                         self.check3[objective], 
                                         self.MaxIntOrder[objective] 
                                         #iteration[objective]
                                         ) 
                      )
            i += 1
        
                
        return y #objective_values
  


