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

from pyANOVAMOP.Metamodelling import SurrogatePrediction

import numpy as np

class ANOVAMOPtest1SubpSurrogate(baseProblem):
    """
      New problem description.
    """
        
    
    def objectives(self, 
                   decision_variables, # a sample
                   SubProblemObjectiveIndices, 
                   P,
                   md,
                   check3,
                   MaxIntOrder
                   ) -> list:
        """Use this method to calculate objective functions.
        Args:
            decision_variables: a sample  
        """
        
        x = decision_variables
        
        NumObj = len(SubProblemObjectiveIndices) # e.g. 3
        #NumVar = len(SubProblemVariablesIndices) # e.g. 3
               
        y = np.zeros(NumObj)
        i = 0
    
        for objective in SubProblemObjectiveIndices: # range(NumObj) 
            y[i] = float(SurrogatePrediction(np.matrix(x), 
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
            i += 1
        
                
        return y #objective_values
  
name = "ANOVAMOPtest1SubpSurrogate"
#k = 10
numobj = 5
numconst = 0
numvar = 5
problem = ANOVAMOPtest1SubpSurrogate(name, numvar, numobj, numconst, )

lattice_resolution = 4
population_size = 105

pop = Population(problem)

pop.evolve(NSGAIII)

pop.non_dominated()

refpoint = 2
volume = 2 ** numobj
#print(pop.hypervolume(refpoint) / volume)



