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
    global SubProblemObjectiveIndices, SubProblemVariablesIndices, P, md, check3, MaxIntOrder 
    
    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions.
        Args:
            decision_variables: a sample  
        """
        
        x = decision_variables
        
        #NumObj = len(SubProblemObjectiveIndices) # e.g. 3
        #NumVar = len(SubProblemVariablesIndices) # e.g. 3
               
        y = np.zeros(len(SubProblemObjectiveIndices))
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
numobj = len(SubProblemObjectiveIndices)
numconst = 0
numvar = len(SubProblemVariablesIndices)

problem = ANOVAMOPtest1SubpSurrogate(name, numvar, numobj, numconst, )

lattice_resolution = 4
population_size = 105

pop = Population(problem)

# You can choose the solver from RVEA and NSGAIII
pop.evolve(NSGAIII)

pop.non_dominated()

refpoint = 2
volume = 2 ** numobj
#print(pop.hypervolume(refpoint) / volume)



