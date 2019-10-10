# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:59:13 2019

@author: babshava
"""


import numpy as np

from pyrvea.Population.Population import Population
#from pyrvea.Problem.baseproblem import BaseProblem
from pyrvea.EAs.RVEA import RVEA
#from pyrvea.EAs.slowRVEA import slowRVEA
from pyrvea.EAs.NSGAIII import NSGAIII
from optproblems import dtlz


class BaseProblem:
    """Base class for the problems."""

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):
        """
        Pydocstring is ruthless.
        Parameters
        ----------
        name
        num_of_variables
        num_of_objectives
        num_of_constraints
        upper_limits
        lower_limits
        """
        self.name = name
        self.num_of_variables = num_of_variables
        self.num_of_objectives = num_of_objectives
        self.num_of_constraints = num_of_constraints
        self.obj_func = []
        self.upper_limits = upper_limits
        self.lower_limits = lower_limits
        self.minimize = None

    def objectives(self, decision_variables):
        """Accept a sample. Return Objective values.
        Parameters
        ----------
        decision_variables
        """
        pass

    def constraints(self, decision_variables, objective_variables):
        """Accept a sample and/or corresponding objective values.
        Parameters
        ----------
        decision_variables
        objective_variables
        """
        pass

    def update(self):
        """Update the problem based on new information."""
        pass

class ANOVAMOPtest1(BaseProblem):
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
problem = ANOVAMOPtest1(name, numvar, numobj, numconst)

lattice_resolution = 4
population_size = 105

pop = Population(problem)
#pop = Population(
#    problem,
#    crossover_type="simulated_binary_crossover",
#    mutation_type="bounded_polynomial_mutation",
#)

pop.evolve(RVEA)

non_dom_index = pop.non_dominated() 
                
xParetoTemp = pop.individuals[non_dom_index[0]]
xsize = np.shape(xParetoTemp)
                
fParetoTemp = pop.fitness[non_dom_index[0]]
xsize = np.shape(xParetoTemp)

#refpoint = 2
#volume = 2 ** numobj
#print(pop.hypervolume(refpoint) / volume)

