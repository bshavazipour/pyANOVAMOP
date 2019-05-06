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



"""
Some predefined functions:
"""

def BuildMdelta(k, d, delta, SM):
    """" Building the reduced incidence matrix Mδ with  ------------------------------
                 Mδ := [m_i^l]_{l,i} s.t \{ m_i^l = 1; if T_i^l >= C
                                         \{ m_i^l = 0; if T_i^l < δ
         where k is the number of objective functons, d is the number of variables and δ is threshold. 
    """
    Mdelta = np.zeros((k,d))
    for l in range(k):
        for i in range(d):
            if SM[l][i] >= delta:
                Mdelta[l][i] = 1
            else:
                Mdelta[l][i] = 0
    return Mdelta
#---------------------------------------------------------------------------


"""
Input: Let f : [0, 1]^d → R^k be a black box vector function in (1). 
"""
d=5 #Number of variables
k=5 #Number of objective functons


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

  1. The dataset, 
  2. Builds the metamodels and,
  3. Estimates the total sensitivity indices.
 """
 
 
  
  
"""
2: Perform the anova analysis on f and build the k × d sensitivity matrix SM.
"""


SM = np.array([[],    # SM will calculate with information from BPC method in 1
               [],
               ...,
               []])

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

w = SM.max(1).min()

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
    # is SM(M) is reducible
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
    16
    17
    18
    """
    #18 if p1m is a list of lists includes p1, p2, ..., pm are the Pareto sets of the subproblems; repectively.
    P = Cartesian_Product_of_m(p1m) # Pareto optimal set for the original problem
    # Similarly, if f1m is a list of lists includes f1, f2, ..., fm are the Pareto frontier approximations of the subproblems; repectively.
    F = Cartesian_Product_of_m(f1m) # Pareto optimal frontier for the original problem
    
elif Reducible:
    # Remove the column of zero from the Mδ
    ReducedMdelta = np.delete(Mdelta,np.where(~Mdelta.any(axis=0))[0], axis=1) #remove the columns that contain only 0
    RemovedColumn = np.where(~Mdelta.any(axis=0))[0] + 1 # Shows the number of column which is removed
    print("The variable which had no effect is x_", int(RemovedColumn))
    
    #(20) solve the reduced problem and return P and F
    
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






"""
25: return  (P , F , V, eps).  
"""






"""
Draft: -----------------------------------------------------------------------------------------
"""            
# Copy the elements of SM to a new created Matrix Mdelta since we want to keep the original SM
Mdelta = np.zeros((k,d))
for l in range(k):
    for i in range(d):
         Mdelta[l][i] = SM[l][i]

# Check if Mδ is reducible i.e. if Mδ has a full column of zeroes 
        
ReduceCheck = (~Mdelta.any(axis=0)).any()  # Return True if find any column of zeroes

np.where(~Mdelta.any(axis=0))[0] # Return the column index if needed
            
#

 

    




        