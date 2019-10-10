# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:11:23 2019

@author: babshava
"""

from pyrvea.Population.Population import Population
from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.NSGAIII import NSGAIII

import numpy as np

class ANOVAMOPtest1Subp(BaseProblem):
    """
      New problem description.
    """
    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
        SubProblemObjectiveIndices = [],
    ):    
    
        self.SubProblemObjectiveIndices = []
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
        NumObj = len(self.SubProblemObjectiveIndices) # e.g. 3
        #NumVar = len(SubProblemVariablesIndices) # e.g. 3
        
        numSample = 1 #, self.num_of_variables = np.shape(np.matrix(x))
        
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
              
        Output = np.zeros(NumObj)
        i = 0
    
        for objective in self.SubProblemObjectiveIndices: # range(NumObj) 
            
            if objective == 0:
                Output[i] = Phi1 + epsilon * Phi4
                
            if objective == 1:
                Output[i] = Phi2 + epsilon * Phi5
                
            if objective == 2:
                Output[i] = Phi3 + epsilon * (Phi4 + Phi5)
                   
            if objective == 3:
                Output[i] = Phi4 + epsilon * Phi1
        
            if objective == 4:
                Output[i] = Phi5 + epsilon * (Phi1 + Phi2)
        
            i += 1
                  
            
        return Output #objective_values
  
name = "ANOVAMOPtest1Subp"
#k = 10
numobj = 5
numconst = 0
numvar = 5

ANOVAMOPtest1Subp.SubProblemObjectiveIndices = [] # it should assign in ANOVAMOP(main) before calling this class

problem = ANOVAMOPtest1Subp(name, numvar, numobj, numconst, )

lattice_resolution = 4
population_size = 105

pop = Population(problem)

pop.evolve(NSGAIII)

non_dom_index = pop.non_dominated() 
                
xParetoTemp = pop.individuals[non_dom_index[0]]
xsize = np.shape(xParetoTemp)
                
fParetoTemp = pop.fitness[non_dom_index[0]]
xsize = np.shape(xParetoTemp)

#refpoint = 2
#volume = 2 ** numobj
#print(pop.hypervolume(refpoint) / volume)
