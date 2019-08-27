# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:33:07 2019

@author: babshava

This is a Python implementation of ANOVAMOP 

Original paper: 
    Tabatabaei, Mohammad, et al. "ANOVA-MOP: ANOVA Decomposition for Multiobjective Optimization."
    SIAM Journal on Optimization 28.4 (2018): 3260-3289.

This impletation has been written as a tutorial for the impelementation of the ANOVAMOP algorithm proposed in the above article (on page 3279)  

"""


import numpy as np
import random
from numpy import linalg as LA
from scipy.optimize import fsolve
from scipy.optimize import minimize

from pyANOVAMOP.CommonFiles import BuildMdelta
from pyANOVAMOP.CommonFiles.Cartesian_product import Cartesian_Product
from pyANOVAMOP.CommonFiles import MyFun
from pyANOVAMOP.CommonFiles import P_objective
from pyANOVAMOP.CommonFiles import SubProblem
from pyANOVAMOP.Decision.making.ASF2 import ASF
from pyANOVAMOP.Decomposition import CheckDecomposibility
from pyANOVAMOP.Decomposition import connectedComponents
from pyANOVAMOP.Metamodelling import BPCfunction


"""
Initialization>

Input: Let f : [0, 1]^d → R^k be a black box vector function in (1). 
"""
d=5 #Number of variables
k=5 #Number of objective functons

ProbName='ALBERTO'# Problem's name defined in P_objective.py


DM = 1# 0 means non-interactive setting and 1 means interactive setting

lb = - np.ones((1,d)) # Lower bounds of variables
ub = np.ones((1,d)) # Upper bound of variables
NumberSubProblems = 2  # Preferred number of subproblems to be obtained by decomposition  # Not used yet
VariableIndices = np.arange(0,d) #np.arange(1,d+1)

"""
The user has an option of selecting a threshold value δ.
"""
# Ask the user to select a threshold value δ
delta = 0
deltachosen  = input("Do you want to select a threshold value (delta)? (Y/N)")
if deltachosen == 'Y' or deltachosen == 'y' or deltachosen == 'Yes' or deltachosen == 'yes' or deltachosen == 'YES':
    delta = float(input("Please enter a threshold value (0 < delta <= 1):"))
    while delta <= 0 or  delta > 1:
        print ("Sorry, that was an invalid value!")
        delta = float(input("Please re-enter a threshold value (0 < delta <= 1):"))
        print("You have decided to select a value for threshold (delta), therefore, delta has been set to:", delta)
elif deltachosen == 'N' or deltachosen == 'n' or deltachosen == 'No' or deltachosen == 'no' or deltachosen == 'NO': 
    print("You have decided to not select any value for threshold (delta), then it will be chosen automatically by the algorithm if needed.")
else:
    print("You have decided to not select any value for threshold (delta), then it will be chosen automatically by the algorithm if needed.")


"""
1: Find a training dataset, apply a metamodeling technique to build 
the metamodels f~ for f and estimate the total sensitivity indices. 

Alternatively, employ BPC [58] (Matthias Tan's method), which ﬁnds 

  1. The dataset (D), 
  2. Builds the metamodels () and,
  3. Estimates the total sensitivity indices (TotalIndices).
 """

SurrogateDataInfo = [[[None]] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
TotalSenIndMatrix = [d * [None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
DataSets = [[[None]] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
Y = [[None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
P = [[None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
md = [[None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
check3 = [[None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
MaxIntOrder = [[None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it
iteration = [[None] for i in range(k)] # Generate an empty 2D k*d list of lists, k and d are the number of objectives and variables,respectively. check if need to increase it

for objective in range(k):
    print(objective)
    SurrogateDataInfo[objective][0] = BPC(ProbName,objective,k,lb,ub,10000)   
    TotalSenIndMatrix[objective][:] = SurrogateDataInfo[objective][0].TotalIndices[:,1] #.T
    DataSets[objective][0] = SurrogateDataInfo[objective][0].D #
    Y[objective] = SurrogateDataInfo[objective][0].Y
    P[objective] = SurrogateDataInfo[objective][0].P
    md[objective] = SurrogateDataInfo[objective][0].md #
    check3[objective] = SurrogateDataInfo[objective][0].check3
    MaxIntOrder[objective] = SurrogateDataInfo[objective][0].MaxIntOrder
    iteration[objective] = SurrogateDataInfo[objective][0].iteration
        

   
"""
2: Perform the anova analysis on f and build the k × d sensitivity matrix SM.



SM = np.array([[],    # SM will calculate with information from BPC method in 1
               [],
               ...,
               []])
"""

SM = TotalSenIndMatrix # The output of the BPC

"""
 e.g from the paper (p 3280-3281):
 
 SM = np.array([[.333, .333, .333, .001, .001],
                [.333, .333, .333, .001, .001],
                [.333, .333, .333, .001, .001],
                [.001, .001, .001, .499, .499],
                [.001, .001, .001, .499, .499]])
"""

"""
# initial check for reducability or decomposability of the SM (M) - before using delta
"""
Mdelta = np.array(SM[:]) # Initialize Mdelta as SM or M
# Check if SM(M) is decomposable
CheckD = CheckDecomposability(d, k, Mdelta)
cc = CheckD[0]  # cc is a list of lists includes the  connected components of the graph
Decomposable = CheckD[1] #  True if the matrix is decomposable and False if not
reMdelta = CheckD[2] # re-ordered matrix Mδ is the second return of function CheckDecomposability(d, k, Mdelta)

# is SM(M) is reducible
Reducible = (~Mdelta.any(axis=0)).any()  # True if the matrix is reducible and False if not

"""
 3: Deﬁne a sorted list E = {e1, . . . , e(k.d)} of 
 all the entries of SM in an increasing order.
"""
 
#E = np.unique(SM) #The number of elements in E (er) is equal to k×d at the most; as the frequent elements are eliminated [use in Matlab code, but bring problems in some specific cases]
E = np.sort(SM, axis=None) #The number of elements in E (er) is equal to k×d, the frequent elements are not eliminated

"""
4: Determine ω according to (11) and ﬁnd the maximum r such that er ≤ ω. 

     ω = Min_{l=1,..,k} [Max_{i=1,..,d} (T_i^l)]
     
  ⊲Here ω is also a valid value for δ because the entries equal to δ remain active.
"""
# Assigning an upper bound for the threshold  ω

w = Mdelta.max(1).min() #SM.max(1).min()

maxnumthreshold = 0 # The maximum r such that er ≤ ω
Thresholds = []  # 
for er in E:
    if er <= w:
        Thresholds.append(er)
        maxnumthreshold += 1 # Alternatively, possible to use len(Thresholds) if it is not used anywhere else
        
"""
5: Extract the list Δ by picking the elements of E from the (d+1)th to ω, i.e., Δ := {ed+1, . . . , er}.
 d = the number of variables
⊲We need at least d entries equal to zero for having reducibility and d+k−2 for decomposability.
"""
Delta = []
for j in range(d,maxnumthreshold):  # Note: d means d+1 since array counter begin from '0'
    Delta.append(Thresholds[j])  # The list Δ := {ed+1, . . . , er}

Delta = list(set(Delta)) # Remove the frequent elements from the list Δ
Delta.sort()  
      
"""
 If the problem is not decomposable; i.e. the incidence matrix M is not sparse enough 
 (as it occur in the most of real-life problems), Then we need and approximated problem 
 which is δ-decomposable or δ-reducible.

To reach this purpose, we need to build a reduced incidence matrix Mδ:
    
6: if δ has been chosen by the user then
    7: Build the reduced incidence matrix Mδ.
    
          Mδ := [m_i^l]_{l,i} s.t \{ m_i^l = 1; if T_i^l >= δ
                                  \{ m_i^l = 0; if T_i^l < δ
    
8: else ⊲Pick the smallest value of δ for which Mδ is reducible or decomposable, if existing.
    9: repeat
    10: Pick as tolerance value δ the smallest element in Δ.
    11: Build the reduced incidence matrix Mδ.
    12: Remove δ from Δ.
    13: until Mδ is reducible or decomposable or Δ is empty.
14: end if
"""       

        
if delta > 0 and  not Reducible and not Decomposable : # 6: It means that user has been selected a value for threshold (delta) 
    Mdelta = BuildMdelta(k, d, delta, SM) # 7: Build the reduced incidence matrix Mδ with a predefined function BuildMdelta(k, d, delta, Mdelta)
    # Check if SM(M) is decomposable
    CheckD = CheckDecomposability(d, k, Mdelta)
    cc = CheckD[0]  # cc is a list of lists includes the  connected components of the graph
    Decomposable = CheckD[1] #  True if the matrix is decomposable and False if not
    reMdelta = CheckD[2] # re-ordered matrix Mδ is the second return of function CheckDecomposability(d, k, Mdelta)
    # if SM(M) is reducible
    Reducible = (~Mdelta.any(axis=0)).any()  # True if the matrix is reducible and False if not

else:  # i.e. if delta == 0 -> it means that delta was not selected by the user, then it needs to be chosen automatically
    # Check if Δ is empthy; Or
    # Check if Mδ is reducible i.e. if Mδ has a full column of zeroes; Or
    # Check if Mδ is decomposable (using function: CheckDecomposability(d, k, Mdelta))
    while len(Delta) != 0 and not Reducible and  not Decomposable:
        delta = Delta[0]  # 10: δ = min(e_i in Δ) [since Delta is sorted in an increasing order, then, the fist element of it is the minimum] 
        Mdelta = BuildMdelta(k, d, delta, SM) # 11: Build the reduced incidence matrix Mδ with a predefined function BuildMdelta(k, d, delta, Mdelta)
        del Delta[0]  # 12: Remove δ from Δ
        CheckD = CheckDecomposability(d, k, Mdelta)
        cc = CheckD[0]  # cc is a list of lists includes the  connected components of the graph
    # The problem is δ-decomposable if the graph has two or more connected components 
    #(i.e. the corresponding matrix, at least, has two blocks).
    # cc is the first return of function CheckDecomposability(d, k, Mdelta)
        Decomposable = CheckD[1] #  True if the matrix is decomposable and False if not
        reMdelta = CheckD[2] # re-ordered matrix Mδ is the second return of function CheckDecomposability(d, k, Mdelta)
        Reducible = (~Mdelta.any(axis=0)).any()  # True if the matrix is reducible and False if not

"""
15: if Mδ is decomposable then
    16: Decompose problem with fδ in approximated subproblems with f^(1), . . . , f^(m) as described in Remark 6.
    17: Solve the subproblems separately by calling solver, which returns the sets of Pareto optima
                    P^(1), . . . , P^(m) and corresponding Pareto frontiers F^(1), . . . , F^(m)
    18: Build the Pareto optimal set approximation for (1), P := P (1) × · · · × P (m),

           P = {(x_1^(1), . . . , x_{d1}^(1), x_1^(2), . . . , x_{d(m−1)}^(m−1), x_1^(m), . . . , x_{dm}^(m)) | 

                                                                           x^(1) ∈ P^(1), . . . , x^(m) ∈ P^(m)}

          and, analogously, the corresponding Pareto frontier approximation F := F^(1) × · · · × F^(m).
    
    

19: else if Mδ is reducible then
    20: Solve the approximated reduced problem fδ by calling solver, which returns the sets of solutions P and F.


21: else
    22: Set P = F = ∅. ⊲The method is not applicable

23: end if
"""

if Decomposable: # 15: If Mδ is decomposable
    """
    16 ->  17  ->  18
    
    #NumberSubProblems = cc # Number of components/blocks found by ANOVA decomposition in reMdelta
    
    16:  Decompose problem with fδ in approximated subproblems with f^(1), . . . , f^(m) as described in Remark 6.
    
    """
     
    fdelta = SubProblems(k, d, cc)
    
    
    """
    17: Solve subproblems separately by calling the solver and get Pareto optimal sets p1, p2, ..., pm
    
    ----------------------------

    Solving subproblems 

        1. Non-interactive 
    
    -----------------------------"""

    if ~DM: #% ANOVA-MOP as a non-interactive method 
                   
        VariableBounds = np.vstack((lb,ub))
        #SubProblemObjectiveIndicesTemp=[]
        #SubProblemVariablesIndicesTemp=[]
    
        for SubProbInd in range(NumberSubProblems): # Solving subproblems one by one
            # Active objectives 
            SubProblemObjectiveIndices = fdelta[SubProbInd][0]
            #SubProblemObjectiveIndicesTemp = np.hstack((SubProblemObjectiveIndicesTemp, SubProblemObjectiveIndices))
            # Active variables
            SubProblemVariablesIndices = np.hstack(([], fdelta[SubProbInd][1]))
            #SubProblemVariablesIndicesTemp = np.hstack((SubProblemVariablesIndicesTemp, SubProblemVariablesIndices))

            if len(SubProblemObjectiveIndices) > 1 and len(SubProblemVariablesIndices) > 1:
                FixedIndices = np.setdiff1d(VariableIndices, SubProblemVariablesIndices)  # Shows the active variables in the sub-problem (in terms of number of columns in the original problem)
                Bounds = VariableBounds[:,SubProblemVariablesIndices.astype(int)]
                FixedValues = VariableBounds[:,FixedIndices].mean(axis=0)
            
                #solve the subproblem and get the solutions [x,f]
                Input, SubInput, Output = SubProblem(SubProblemObjectiveIndices,
                                                      SubProblemVariablesIndices,
                                                      Bounds,
                                                      lb,
                                                      ub,
                                                      FixedIndices,
                                                      FixedValues,
                                                      #model  # model = SurrogateDataInfo has all info about the all objectives returend from the BPC; SurrogateDataInfo[i] has the info of the i-th objectives, e.g. SurrogateDataInfo[i].md  
                                                      DataSets,#[objective][0] 
                                                      #Y[objective] 
                                                      P,#[objective]
                                                      md,#[objective] 
                                                      check3,#[objective] 
                                                      MaxIntOrder#[objective] 
                                                      #iteration[objective] 
                                                      )
    
    
                random.choice(Input)
                
                [xParetoTemp, fParetoTemp] = 
                        
                p1m[SubProbInd] = xParetoTemp # p1m = p1 ... pm  (SubProblemsDecisionSpacePareto)
                f1m[SubProbInd] = fParetoTemp   # f1m = f1 ... fm  (SubProblemsObjectiveSpacePareto)   
         
            
            
            
            """----------------------------

             Solving subproblems 

                2. Interactive 
    
            -----------------------------"""

    else:  # ANOVAMOP as an interactive method
        FinalSolutionDecisionSpace = np.zeros((1,d))  #P
        FinalSolutionObjectiveSpace = np.zeros((1,k)) #F
        
        for SubProbInd in range(NumberSubProblems):
            VariableBounds = np.vstack((lb,ub))
            # Active objectives
            SubProblemObjectiveIndices = fdelta[SubProbInd][0]
            # Active variables
            SubProblemVariablesIndices = np.hstack(([], fdelta[SubProbInd][1]))
            
            FixedIndices = np.setdiff1d(VariableIndices, SubProblemVariablesIndices)
            Bounds = VariableBounds[:,SubProblemVariablesIndices.astype(int)]
            FixedValues = VariableBounds[:,FixedIndices].mean(axis=0)
            VariableBounds[:,FixedIndices] = np.matlib.repmat(FixedValues,2,1) # check if 2 is a correct number
                    
            # Generating the initial samples (Input) for this sub-problem and get the estimated values for active objectives in it (Output)
            
            Input, SubInput, Output = SubProblem(SubProblemObjectiveIndices,
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
                                       )
            
            
            DoInteraction = 1
            # Ask the DM to set relevant aspiration levels
            while DoInteraction:
                print("Indices of objective functions of subproblem #", str(SubProbInd+1), " are ", str(np.matrix(SubProblemObjectiveIndices)+1), ".")
                print('Enter a referenc point as a vector 1 *', str(len(SubProblemObjectiveIndices)), ' for these objectives ')  #in a list like [0, 0, ..., 0]
                z = input('') # Reference points provided by the DM
                
                if len(z) != len(SubProblemObjectiveIndices): # needs modification in Input()
                    print('The reference point must have a dimension 1 *', str(len(SubProblemObjectiveIndices)), '.')
                    
                #elif type(z) != list:
                #    print('The reference point must be a list like [0, 0, ..., 0].')
                
                                       
                else:
                                            
                    """
                    Solve the subproblem and get the solutions [x,f],
                    Show solutions to the DM and,
                    Ask to change the reference points if (s)he is not happy with this solution 
                    """
                    xOptApp, fOptApp = MakeDecision(z, # z must be a list like [0,0,...,0]
                                                    #fdelta[SubProbInd], # related to current sub-problem
                                                    #SurrogateDataInfo,
                                                    #DataSets,#[objective][0] 
                                                    #Y[objective] 
                                                    P,#[objective]
                                                    md,#[objective] 
                                                    check3,#[objective] 
                                                    MaxIntOrder,#[objective] 
                                                    #iteration[objective]
                                                    SubProblemObjectiveIndices,
                                                    #VariableBounds,
                                                    Input
                                                    )
                    
                    print("Given reference point = [", str(z), "], and ")
                    print("corresponding prefered solution in the objecive space = [", str(fOptApp), "].")
                    Tag = input('Would you like to provide another reference point? (Yes = 1 / No = 0) ')
                        
                    if Tag == 0:
                        DoInteraction = 0
                        FinalSolutionDecisionSpace[SubProblemVariablesIndices] = xOptApp[SubProblemVariablesIndices]
                        FinalSolutionObjectiveSpace[SubProblemObjectiveIndices] = fOptApp
                
                            
                            
            #save FinalSolutionDecisionSpace FinalSolutionDecisionSpace
            #save FinalSolutionObjectiveSpace FinalSolutionObjectiveSpace
            #print('The final solution in the objective space is')
            #print('          [' num2str(FinalSolutionObjectiveSpace) ']')
            #print('The final solution in the decision space is')
            #print('          [' num2str(FinalSolutionDecisionSpace) ']')

    
    
    """
    18: Generate the Pareto optimal set & Pareto frontier for the original problem
    
      if p1m is a list of lists includes p1, p2, ..., pm are the Pareto sets of the subproblems; repectively.
    """
    P = Cartesian_Product_of_m(p1m) # Pareto optimal set for the original problem
    # Similarly, if f1m is a list of lists includes f1, f2, ..., fm are the Pareto frontier approximations of the subproblems; repectively.
    F = Cartesian_Product_of_m(f1m) # Pareto optimal frontier for the original problem
    
elif Reducible:
    # Remove the column of zero from the Mδ
    ReducedMdelta = np.delete(Mdelta,np.where(~Mdelta.any(axis=0))[0], axis=1) #remove the columns that contain only 0
    RemovedColumn = np.where(~Mdelta.any(axis=0))[0] + 1 # Shows the number of column which is removed
    print("The variable which had no effect is x_", int(RemovedColumn))
    
    #(20) solve the reduced problem and return P and F
    -built the new sub problem
    -solve it and get P and F
    
    P =     # Pareto optimal set for the original problem
    F =     # Pareto optimal frontier for the original problem
    
else: # (21) if Δ is empthy
    P = []
    F = []
    print("There are some common variables between subproblems, then the ANOVA-MOP cannot solve such problems.")



"""
24: Select randomly a validation sample V := {x^(1), . . . , x^(v)}	 ⊆ P and estimate the approximation

quality eps of the solutions found as follows:
    

    (16) eps ≃ max_{x∈V}  || f(x) − fδ(x) ||_∞ = max_ {ν=1,...,v}   max_{ℓ=1,...,k} | 
f^ℓ (x^(ν)) − fδ^ℓ (x(ν)) |
.
 
"""
# Select randomly a validation sample V := {x^(1), . . . , x^(v)}	 ⊆ P
V = random.choices(P,k=random.randint(1,len(P)))

# Estimate the approximation quality eps of the solutions found
#for v in V:
#    max(max(abs(fl(xv) − fδl(xv))))
eps = LA.norm(F(xv) − Fδ(xv), ord=np.inf)

"""
25: return  (P , F , V, eps).  
"""
#return  (P , F , V, eps)







"""

"""

def MakeDecision(z,
                 #DecomposedModel, # fdelta[SubProbInd]
                 #SurrogateDataInfo,
                 #DataSets,#[objective][0] 
                 #Y[objective] 
                 P,#[objective]
                 md,#[objective] 
                 check3,#[objective] 
                 MaxIntOrder,#[objective] 
                 #iteration[objective]
                 ObjIndices, #SubProblemObjectiveIndices,
                 #DecomposedBounds #VariableBounds
                 Input
                 ):
    """
     
    # Initialization for DIRECT method
    opts.es = 1e-4
    opts.maxevals = 500
    opts.maxits = 500
    opts.maxdeep = 1000
    opts.testflag = 0
    opts.showits = 0
    
    
        
     
    #call an optimizer e.g. minimize or fsolve
    
    # ASF will be called within the optimizer
    #Problem.f ='ASF' # objective function from ASF
   
    
    # Solve the single objective optimizer to Min ASF(X,Z) of the (sub-)problem and get the solution
    #[~, xOptApp] = Direct(Problem,VariableBounds',opts,StrucData)
    """
    
    xOptApp = minimize(ASF, 
                       random.choice(Input), # 
                       args=(#model  # model = SurrogateDataInfo has all info about the all objectives returend from the BPC; SurrogateDataInfo[i] has the info of the i-th objectives, e.g. SurrogateDataInfo[i].md  
                           #DataSets[0][0][0], 
                           #Y[objective] 
                           P,#[objective]
                           md,#[objective] 
                           check3,#[objective] 
                           MaxIntOrder,#[objective] 
                           #iteration[objective] 
                           ObjIndices, #SubProblemObjectiveIndices,
                           #DecomposedBounds,  #VariableBounds
                           z
                           )
                     )
                         
    #xOptApp = xOptApp.T
    
    
    # Function evluations with new solution
    fOptApp = np.zeros(len(ObjIndices))
    #fOptApp = []
    i = 0
    for objective in ObjIndices: #len(ObjIndices) 
        fOptApp[i] = float(SurrogatePrediction(np.matrix(xOptApp.x),
                                               #SurrogateDataInfo[SubProblemObjectiveIndices[Objective]]
                                               #model: # model stored as Data includes e.g. md, check3, P, MaxIntOrder
                                               #DataSets[objective][0], 
                                               #Y[objective] 
                                               P[objective],
                                               md[objective], 
                                               check3[objective], 
                                               MaxIntOrder[objective], 
                                               #iteration[objective]
                                               )
                           )
        i += 1
        
    return (xOptApp.x, fOptApp)






        