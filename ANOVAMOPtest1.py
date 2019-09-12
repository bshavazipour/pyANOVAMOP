# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:59:13 2019

@author: babshava
"""


from pyANOVAMOP.Solvers.pyRVEA.pyrvea.Population.Population import Population
from pyANOVAMOP.Solvers.pyRVEA.pyrvea.Problem.baseProblem import baseProblem
from pyANOVAMOP.Solvers.pyRVEA.pyrvea.EAs.RVEA import RVEA
from pyANOVAMOP.Solvers.pyRVEA.pyrvea.EAs.NSGAIII import NSGAIII
#from optproblems import dtlz
import numpy as np

class ANOVAMOPtest1(baseProblem):
    """
      New problem description.
    """
        
    
    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions.
        Args:
            decision_variables: a sample  
        """
        
        x = decision_variables
        
        numSample, self.num_of_variables = np.shape(np.matrix(x))
        
        epsilon = 0.1
                
        P1 = np.array([1, 1, 1])
        P2 = np.array([1, -1, -1])
        P3 = np.array([1, 1, -1])
        P4 = np.array([1, -1])
        P5 = np.array([-1, 1])
        
        Phi1 = ((x[0:3] - np.ones((numSample,1)) * P1) ** 2).sum(axis=1)
        Phi2 = ((x[0:3] - np.ones((numSample,1)) * P2) ** 2).sum(axis=1)
        Phi3 = ((x[0:3] - np.ones((numSample,1)) * P3) ** 2).sum(axis=1)
        Phi4 = ((x[3:5] - np.ones((numSample,1)) * P4) ** 2).sum(axis=1)
        Phi5 = ((x[3:5] - np.ones((numSample,1)) * P5) ** 2).sum(axis=1)
        
        Output = np.empty((numSample,5))
        Output[:,0] = Phi1 + epsilon * Phi4
        Output[:,1] = Phi2 + epsilon * Phi5
        Output[:,2] = Phi3 + epsilon * (Phi4 + Phi5)
        Output[:,3] = Phi4 + epsilon * Phi1
        Output[:,4] = Phi5 + epsilon * (Phi1 + Phi2)
        
                
        return Output #objective_values
  
name = "ANOVAMOPtest1"
#k = 10
numobj = 5
numconst = 0
numvar = 5
problem = ANOVAMOPtest1(name, numvar, numobj, numconst, )

lattice_resolution = 4
population_size = 105

pop = Population(problem)

pop.evolve(NSGAIII)

pop.non_dominated()

refpoint = 2
volume = 2 ** numobj
#print(pop.hypervolume(refpoint) / volume)

