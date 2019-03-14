# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:33:07 2019

@author: babshava
"""


import numpy as np



"""
Some predefined functions:
"""
# Build the reduced incidence matrix Mδ with  ------------------------------
# Mδ := [m_i^l]_{l,i} s.t \{ m_i^l = 1; if T_i^l >= C
#                         \{ m_i^l = 0; if T_i^l < δ
# where k is the number of objective functons, d is the number of variables and δ is threshold. 

def BuildMdelta(k, d, delta, SM):
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
Input: Let f : [0, 1]d → R^k be a black box vector function in (1). 
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
2: Perform the anova analysis on ef and build the k × d sensitivity matrix SM.
"""


SM = np.array([[],    # SM will calculate with information from BPC method in 1
               [],
               ...,
               []])

# e.g SM = np.array([[23, 549, 1, 48,38],[77, 4, 387, 12, 83], [45, 397, 8, 25, 1]])

"""
 3: Deﬁne a sorted list E = {e1, . . . , e(k.d)} of 
 all the entries of SM in an increasing order.
"""
 
E = np.unique(SM) #The number of elements in E (er) is equal to k×d at the most; as the frequent elements are eliminated


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

        
 """
 If the problem is not decomposible; i.e. the incidence matrix M is not sparse enough 
 (as it occur in the most of real-life problems), Then we need and approximated problem 
 which is δ-decomposible or δ-reducible.

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


if delta > 0 : # 6: It means that user has been selected a value for threshold (delta) 
    Mdelta = BuildMdelta(k, d, delta, SM) # 7: Build the reduced incidence matrix Mδ with a predefined function BuildMdelta(k, d, delta, Mdelta)
else:  # i.e. if delta == 0
    # Check if Δ is empthy; Or
    # Check if Mδ is reducible i.e. if Mδ has a full column of zeroes; Or
    # Check if Mδ is decomposible
    while len(Delta) == 0 or (~Mdelta.any(axis=0)).any() or len(cc) >= 2:
        delta = Delta[0]  # 10: δ = min(e_i in Δ) [since Delta is sorted in an increasing order, then, the fist element of it is the minimum] 
        Mdelta = BuildMdelta(k, d, delta, SM) # 11: Build the reduced incidence matrix Mδ with a predefined function BuildMdelta(k, d, delta, Mdelta)
        del Delta[0]  # 12: Remove δ from Δ
        CheckDecomposability(d, k, Mdelta)
        

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

 

    




        