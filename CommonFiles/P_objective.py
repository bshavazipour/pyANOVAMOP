# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:51:38 2019

@author: babshava
"""

"""
Test Problems  ----  P_objective

"""

import math

def P_objective(
        Operation,
        Problem,
        M,
        Input
        ):#,
        #SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model
        #):
    """
        
    """
    
    #k = np.nonzero(~isstrprop(Problem,'digit'),1,'last');
    #switch Problem(1:k)
    if Problem == 'ZDT':
        [Output,Boundary,Coding]=P_ZDT(Operation,Problem,Input)
    elif Problem == 'DTLZ':
        [Output,Boundary,Coding] = P_DTLZ(Operation,Problem,M,Input)
    elif Problem == 'SDTLZ':
        [Output,Boundary,Coding] = P_SDTLZ(Operation,Problem,M,Input)
    elif Problem == 'ALBERTO':
        Output = P_ALBERTO(Operation,Input) # [Output,Boundary,Coding]
    elif Problem == 'Surrogate':
        [Output,Boundary,Coding] = P_Surrogate(Operation,M,Input,SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model)            
    else:
        raise Exception(Problem,'Not Exist'.format(x))
        
     
    return Output#,Boundary,Coding)
    


def P_ALBERTO(Operation,Input):

    """    
    An example in section 4.1 of the ANOVA-MOP paper (p 3280)
    """

    Boundary = math.nan
    Coding = math.nan
    
    #Population Initialization
    if Operation == 'init':
        
        numVar = 5
        Min = -np.ones((1,numVar))
        Max = np.ones((1,numVar))
        Population = np.random.rand(Input,numVar) # Input must be a number # create a random matrix
        #Population=((Input+1).*repmat(range,Input,1))/2+repmat(Min,Input,1);
        Population = Population * np.matlib.repmat(Max,Input,1) + (1 - Population) * np.matlib.repmat(Min,Input,1)
        Output = Population
        Boundary = np.array([[Min],[Max]])
        Coding = 'Real'

    if Operation == 'value':
    
        [numSample, numVar] = np.shape(Input)
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
        
        Output = np.empty((numSample,5))
        Output[:,0] = Phi1 + epsilon * Phi4
        Output[:,4] = Phi2 + epsilon * Phi5
        Output[:,2] = Phi3 + epsilon * (Phi4 + Phi5)
        Output[:,3] = Phi4 + epsilon * Phi1
        Output[:,1] = Phi5 + epsilon * (Phi1 + Phi2)

    return Output#,Boundary,Coding)




def P_Surrogate(Operation,
                M, # NumObj (e.g. 3)
                Input, # Data.D
                SubProblemObjectiveIndices, # active objtives' numbers in a list; e.g. [0,2,4]
                SubProblemVariablesIndices, # active variables' numbers in a list; e.g. [0,1,2]
                NumVar,
                Bounds,
                lb,
                ub,
                FixedIndices, 
                FixedValues, 
                model):
    
    """
    Surrogate problem, use in solving the subproblems via RVEA
    """
    Boundary = math.nan
    Coding = math.nan
    
    #Population Initialization; randomly generate the initial population- it can be done in RVEA as well 
    if Operation == 'init':
        
        D = NumVar
        MaxValue   = Bounds[0,:]
        MinValue   = Bounds[1,:]
        Population = np.random.rand(Input,D) # Input must be a number # create a random matrix
        Population = Population * np.matlib.repmat(MaxValue,Input,1) + (1 - Population) * np.matlib.repmat(MinValue,Input,1)
        Output = Population
        Boundary = np.array([[MaxValue],[MinValue]])
        Coding = 'Real'

    if Operation == 'value':
        
        NumPop = Input.shape[0]
        InputTemp = np.zeros((NumPop,len(SubProblemVariablesIndices) + len(FixedIndices)))
        InputTemp[:,FixedIndices] = np.matlib.repmat(FixedValues,NumPop,1)
        InputTemp[:,SubProblemVariablesIndices] = Input[:,SubProblemVariablesIndices]
        Input = MapSamples(InputTemp, np.vstack((-np.ones((1,len(lb))), np.ones((1,len(lb))))), np.vstack((lb,ub)))
        Output = np.zeros((NumPop,M))
        
        ¤Output = np.empty((NumPop,len(SubProblemObjectiveIndices)))
        ¤for objective in SubProblemObjectiveIndices:#range(M):
         ¤   Output[:,objective] = SurrogatePrediction(Input,SurrogateDataInfo[objective]) 

    return Output#,Boundary,Coding)





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














#_____________________------------------------------------
   
    
    
    
class baseProblem:
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

    def objectives(self, decision_variables):
        """Accept a sample. Return Objective values.
        Parameters
        ----------
        decision_variables
        """
        pass

    def constraints(self, decision_variables):
        """Accept a sample and/or corresponding objective values.
        Parameters
        ----------
        decision_variables
        """
        pass
    
#--------------------------------------------------------------------------------    
    
    
from optproblems import dtlz, zdt
from pyrvea.Problem.baseProblem import baseProblem


class testProblem(baseProblem):
    """Defines the problem."""

    def __init__(
        self,
        name=None,
        num_of_variables=None,
        num_of_objectives=None,
        num_of_constraints=0,
        upper_limits=1,
        lower_limits=0,
    ):
        """Pydocstring is ruthless.
        Args:
            name:
            num_of_variables:
            num_of_objectives:
            num_of_constraints:
            upper_limits:
            lower_limits:
        """
        super(testProblem, self).__init__(
            name,
            num_of_variables,
            num_of_objectives,
            num_of_constraints,
            upper_limits,
            lower_limits,
        )
        if name == "ZDT1":
            self.obj_func = zdt.ZDT1()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT2":
            self.obj_func = zdt.ZDT2()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT3":
            self.obj_func = zdt.ZDT3()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT4":
            self.obj_func = zdt.ZDT4()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT5":
            self.obj_func = zdt.ZDT5()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "ZDT6":
            self.obj_func = zdt.ZDT6()
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ1":
            self.obj_func = dtlz.DTLZ1(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ2":
            self.obj_func = dtlz.DTLZ2(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ3":
            self.obj_func = dtlz.DTLZ3(num_of_objectives, num_of_variables)
            self.lower_limits = 0
            self.upper_limits = 1
        elif name == "DTLZ4":
            self.obj_func = dtlz.DTLZ4(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ5":
            self.obj_func = dtlz.DTLZ5(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ6":
            self.obj_func = dtlz.DTLZ6(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        elif name == "DTLZ7":
            self.obj_func = dtlz.DTLZ7(num_of_objectives, num_of_variables)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
            
        elif name == "BS1":
            self.obj_func = P_BS1(Operation,Input)
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
            
        elif name == "Surrogate":
            self.obj_func = P_Surrogate((Operation,M,Input, SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model):
            self.lower_limits = self.obj_func.min_bounds
            self.upper_limits = self.obj_func.max_bounds
        

    def objectives(self, decision_variables) -> list:
        """Use this method to calculate objective functions.
        Args:
            decision_variables:
        """
        return self.obj_func(decision_variables)

    def constraints(self, decision_variables, objective_variables):
        """Calculate constraint violation.
        Args:
            decision_variables:
            objective_variables:
        """
        print("Error: Constraints not supported yet.")    
        
#--------------------------------------------------------------------------------    

from pyrvea.Population.Population import Population
from pyrvea.Problem.baseProblem import baseProblem
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.NSGAIII import NSGAIII
from optproblems import dtlz


class newProblem(baseProblem):
    """New problem description."""
    
    #def __init__(
        #self,
        #name = None, # problem name
        #num_of_objectives[float] = 0, # M
        #data = [], # Sample, Input
        #SubProblemObjectiveIndices = [], 
        #SubProblemVariablesIndices = [],
        #num_of_variables = None, # NumVar
        #Sub_problem_lb = [],
        #Sub_problem_ub = [],
        #lower_limits = [],
        #upper_limits = [],
        #FixedIndices = [],
        #FixedValues = [],
        #model = [], # SurrogateDataInfo
        #num_of_constraints = 0, # Not supported yet
             
    #):
        """
        Args:
        name :
        num_of_objectives :
        Input :
        SubProblemObjectiveIndices : 
        SubProblemVariablesIndices :
        num_of_variables :
        Sub_problem_lb :
        Sub_problem_ub :
        lower_limits :
        upper_limits :
        FixedIndices :
        FixedValues :
        model :
        num_of_constraints : Not supported yet
        """
        
        #super().__init__() # call the baseProblem __init__ function
        
        
        #Sub_problem_lb = Bounds[0,:]
        #Sub_problem_ub = Bounds[1,:]
                
        #if name == "P_ALBERTO":
            #self.obj_func = P_ALBERTO(Input)
            #self.lower_limits = self.obj_func.min_bounds
            #self.upper_limits = self.obj_func.max_bounds
            
        #elif name == "P_Surrogate":
            #self.obj_func = P_Surrogate(M,Input, SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model):
           # self.lower_limits = self.obj_func.min_bounds
           # self.upper_limits = self.obj_func.max_bounds
            
        #else:
         #   raise Exception(Problem,'Not Exist'.format(x))
            

    
    
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
  
name = "P_ALBERTO"
#k = 10
numobj = 5
numconst = 0
numvar = 5
problem = newProblem(name, numvar, numobj, numconst, )

lattice_resolution = 4
population_size = 105

pop = Population(problem)

pop.evolve(RVEA)

pop.non_dominated()

refpoint = 2
volume = 2 ** numobj
#print(pop.hypervolume(refpoint) / volume)



#------------------------------


    def P_ALBERTO(self, decision_variables):

        """    
            An example in section 4.1 of the ANOVA-MOP paper (p 3280)
        """
       
        x = decision_variables
        
        numSample, self.num_of_variables = np.shape(decision_variables)
        
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
        
        Output = np.empty((numSample,5))
        Output[:,0] = Phi1 + epsilon * Phi4
        Output[:,4] = Phi2 + epsilon * Phi5
        Output[:,2] = Phi3 + epsilon * (Phi4 + Phi5)
        Output[:,3] = Phi4 + epsilon * Phi1
        Output[:,1] = Phi5 + epsilon * (Phi1 + Phi2)

        return Output



    def P_Surrogate(self, M,Input, SubProblemObjectiveIndices,SubProblemVariablesIndices,NumVar,Bounds,lb,ub,FixedIndices, FixedValues, model):
    
        """
        Surrogate problem, use in solving the subproblems via RVEA
        """
                           
        # MaxValue   = Bounds[0,:]
        # MinValue   = Bounds[1,:]
             
        NumPop = Input.shape[0]
        InputTemp = np.zeros((NumPop,len(SubProblemVariablesIndices) + len(FixedIndices)))
        InputTemp[:,FixedIndices] = np.matlib.repmat(FixedValues,NumPop,1)
        InputTemp[:,SubProblemVariablesIndices] = Input
        Input = MapSamples(InputTemp, np.vstack((-np.ones((1,len(lb))), np.ones((1,len(lb))))), np.vstack((lb,ub)))
        Output = np.zeros((NumPop,M))
        
        for objective in range(M):
            Output[:,objective] = SurrogatePrediction(Input,SurrogateDataInfo[SubProblemObjectiveIndices[objective]])

        return Output
    
    
    
    def SurrogatePrediction(x0, model): # model stored as Data includes e.g. md, check3, P, MaxIntOrder
        """
        Evaluate the objective functions for the given solution x0
        """
        md = model.md
        check3 = model.check3
        P = model.P
        MaxIntOrder = model.MaxIntOrder
        x0 = MultivariateLegendre2(x0,P,MaxIntOrder)
        x0 = x0[:,check3]
        Pred = np.matmul(x0,md) # check if the right product has been used here
    
        return Pred


    
    
    
    
    
    
    
    



    


#------------------------
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
