# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:40:55 2019

@author: babshava
"""

"""
This is a Python implementation of the method developed in 
 "M. Tan. Sequential Bayesian polynomial chaos model selection for estimation of
 sensitivity indices. SIAM/ASA Journal on Uncertainty Quantification, 3:146–168, 2015."
 
 by Babooshka Shavazipour
 
 For any question regarding this code, please contact babooshka.b.shavazipour@jyu.fi
 
"""

"""
Sequential process algorithm (Figure 1. p155 in the above paper)
"""
import numpy as np
import pandas as pd
import itertools
import math

import sobol_seq

from scipy.spatial.distance import cdist  # To calculatr thr euclidean distance between tow sets of observations in a matrix form

#from scipy.sparse import eye

#import scramble

#import matplotlib
#import joblib
#import pce  # a package for Polynomial Chaos Expansion method
"""
Initial assignments
"""

#Dimension of problem (Number of variables)
d = 5
#Maximum polynomial degree of orthonormal polynomial regressors
P = 5
#Maximum order of ANOVA functional components to allow in regression
MaxIntOrder = d

#Compute initial sample size
p0 = 4

if d > 1:
    n0 = 1 + p0*d + math.ceil((d-1)*(p0-1)*p0/4)
else:
    n0 = 1 + p0
    
#Number of runs to add at each iteration
nadd = 2 * d

#Set tolerances
Tolerance3 = 15
Tolerance1 = 10
Tolerance2 = 15

#Prior parameters in Bayesian model
rho = 0.5
gamma0 = 10**4
gamma = 240
r = 0.6
lrho = math.log(rho)

#Parameters for sequential procedure, h2>=h1>=2
h1 = 2
h2 = 3

#Program only returns information on Sobol indices up to the order given by truncate
truncate = d

"""
 1. First stage in the algorithm
 
   ' Set value of P ' # Already set
   ' Generate initial design of size n=n0 from scrambled Sobol sequence.' 
   #Low discrepancy quasi-random sequences, e.g. Sobol sequences, fill a space more uniformly than uniformly random sequences.
   ' Set k=0. '
"""

#Generate initial design
n = n0
#Matlab code : QuasiMC = sobolset(d)
#              QuasiMC=scramble(QuasiMC,'MatousekAffineOwen')
#              D=net(QuasiMC,n)*2-1
D = scrambled_sobol(d, n) * 2 - 1 # returns the first n points from the Sobol sequence
# remeber, If G_i is the CDF of u_i, then v_i=2 * G(u_i) -1 is uniformly distributed in [-1, 1]. 


QuasiMCResampling = scrambled_sobol(d, n) #Resampling option (Not used in this version)
ReSampleIndex = 1


"""
 2. Run experiment: evaluate function at all points in design. i.e. find Y = f(X); for all samples in D
 
"""

Y = 


"""
 3.1. Generate regressors
"""
# Generate regressors
(X, alpha, AnovaIndicators, Lambda) = MultivariateLegendre(D, P, MaxIntOrder)
Nf = AnovaIndicators.shape[0]
M = X.shape[1]
M2 = M-1

#Interaction order and number of levels for each component of delta
 
E = np.array([sum(row) for row in AnovaIndicators]).T
F = P - E + 2

#Number of Sobol indices to compute
NI = sum(E <= truncate)

"""
3.2. Starting models for global search
"""
if d > 1 and P > 5:   # eq. 27 p 157
    NoStart0 = 2 ** d - 1 + 2 ** d -1 - d
    deltastart0 = np.zeros((NoStart0, Nf))
    deltastart0[0:d,0:d] = P * np.identity(d)
    No = d
    for i in range(2,d+1):
        Factors = np.array(list(itertools.combinations(range(1,d+1), i)))
        NoFactors = Factors.shape[0]
        for j in range(NoFactors):
            Include = np.logical_and(((E-np.array([sum(row) for row in AnovaIndicators[:,Factors[j,:]]]).T) == 0), (E <= 2))
            No += 1
            deltastart0[No-1,Include] = P
            No += 1
            deltastart0[No-1,Include] = 2

elif d >1 and P <= 5:  # eq. 27 and 28 - p 157-158
    NoStart0 = 2 ** d - 1 
    deltastart0 = np.zeros((NoStart0, Nf))
    deltastart0[0:d,0:d] = P * np.identity(d)
    No = d
    for i in range(2,d+1):
        Factors = np.array(list(itertools.combinations(range(1,d+1), i)))
        NoFactors = Factors.shape[0]
        for j in range(NoFactors):
            Include = np.logical_and(((E-np.array([sum(row) for row in AnovaIndicators[:,Factors[j,:]-1]]).T) == 0), (E <= 2))
            No += 1
            deltastart0[No-1,Include] = P

elif d == 1:
    deltastart0 = np.arange(P+1)[np.newaxis].T
    NoStart0 = P+1



"""
Initialize some variables
"""
iteration = 0
stop = 0
StoremdMinusEta = np.zeros((1,M))
BestModel = []
while stop = 0:
    
    if iteration > 0: 
        # Add the best model found in the previous iteration to the set of starting models
        if  min(cdist(BestModel, deltastart0)[iteration-1]) > 0.5:
            deltastart = np.vstack((deltastart0, BestModel[iteration-1]))
            NoStart = NoStart0 + 1
    else:
        deltastart = deltastart0
        NoStart = NoStart0
            
    if n > n0:
        XAdd = MultivariateLegendre2(D[(n-nadd):n,:],P,MaxIntOrder)
        X = np.vstack((X, XAdd))
        
    #Compute the centered response used in model selection
    mY = Y.mean(0) # Y must be array or matrix (np.array or np.matrix)
    Y2 = np.matrix(Y) - np.matrix(X)[:,0]*mY
    
    #Precompute some matrices to speed up computation of Q(delta)
    W = X[:,1:M]
    W2W2T0 = gamma0 * np.matrix(X)[:,0] * np.matrix(X)[:,0].T    
    
    for i in range(M2):
        W[:,i] = W[:,i] * math.sqrt(gamma * r ** (Lambda[1,i]-1))
    
    R00 = W2W2T0 + np.identity(n)    
        
    """
    4. Global Search - p157-159
    
    Modiﬁed Fedorov algorithm is utilized instead of the Fedorov algorithm (Cook and Nachtsheim, 1980) employed by Tan and Wu (2013).
    """
    deltaopt = np.zeros((NoStart,Nf))
    Q = np.zeros((NoStart,1))
    print('Total number of starting points is ', NoStart, '.')
    
    for i in range(NoStart):
      ¤  [deltaopt[i,:], Q[i]] = Search(deltastart[i,:]) # Search need to be define than test this part
        print('Objective =', ObjInd, '. Function evaluation = ', n, '. Iteration =', iteration,'. Global search from ', i, '/', NoStart,' starting points completed.')
       
    iteration += iteration
    [MinQ, IndexMinQ] = min(Q) ¤ needs to be checked after writing the Search func
    BestModel.append(deltaopt[IndexMinQ,:])
    
    print('Best model found')
    print(np.array(BestModel)[iteration,:])   
    
    #Precompute quantities needed in computing credible limits
    delta = np.array(BestModel)[iteration,:]
   ¤ check = Lambda[1,:] <= delta(Lambda[0,:])
    check2 = np.where(check)
    check3 = [1, check2+1]
    R = R00 + np.matrix(W)[:,check2] * np.matrix(W)[:,check2].T
    IR = invandlogdet(R)
    S = [gamma0, gamma*(r**(Lambda[1,:]-1))]
   ¤ S = np.diag(S(check3));
   ¤ XS = X(:,check3) * S
    Gd = S - (XS.T) * IR * XS
    RSS = (Y2.T) * IR * Y2
    Sigma2Est = RSS / n
   ¤ md = [mY; zeros(length(check2),1)]+XS'*IR*Y2;
  ¤  StoremdMinusEta(iteration,check3)=md-[mY; zeros(length(check2),1)];
   ¤ weights=1./diag(IR);
   ¤ PredCriterion1(iteration,1)=sqrt(mean((weights.*(Y-X(:,check3)*md)).^2))/std(Y,1)*100;
    print('RMS of leave-one-out prediction errors/Standard deviation of Y (in percent)')
    print(PredCriterion1(iteration,1))

    """
    5. Check if stopping conditions have been met
    """









