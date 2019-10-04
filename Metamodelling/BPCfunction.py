# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:10:29 2019

@author: babshava
"""

"""

BPC as a function

"""
import numpy as np
#import pandas as pd
import itertools
import math
#from scipy import stats
#import sobol_seq

from scipy.spatial.distance import cdist  # To calculatr thr euclidean distance between tow sets of observations in a matrix form
from pyANOVAMOP.Metamodelling.ScrambledSobolSequence import scrambled_sobol # scramble,
from pyANOVAMOP.Metamodelling.MultivariateLegendre import MultivariateLegendre2, MultivariateLegendre # ,orthonormal_polynomial_legendre,
from pyANOVAMOP.Metamodelling.Some_required_functions_for_BPCmain import Pred, Search, invandlogdet #, calculatem2lprob, calculatem2lprob2, 
from pyANOVAMOP.Metamodelling.Some_required_functions_for_BPCmain import SimulateSobolIndices #, CheckIfPosDef, CdfDev, CDFEst, PiecewiseLinearCDF, ecdf
from pyANOVAMOP.CommonFiles.MyFun import MyFun

#from scipy.sparse import eye

#import scramble

#import matplotlib
#import joblib
#import pce  # a package for Polynomial Chaos Expansion method

#----------------------------------------

class Data(object):
    """
    To save data from BPC
    """
    def __init__(self, D, Y, Pd, md, check3, MaxIntOrder, iteration, TotalIndices):
        self.D = D
        self.Y = Y
        self.Pd = Pd
        self.md = md
        self.check3 = check3
        self.MaxIntOrder = MaxIntOrder
        self.iteration = iteration
        self.TotalIndices = TotalIndices
        
#-----------------------------------------------------
  

def BPC(ObjInd,lb,ub,MaxNumFunEval, ProblemName,d,k):
    """
    description
    
    """
          
    """
    Initial assignments
    """

    #Dimension of problem (Number of variables)
    #d = 5
    #k = ObjNum # Number of objectives
    #Maximum polynomial degree of orthonormal polynomial regressors
    Pd = 5 # originally was 5
    #Maximum order of ANOVA functional components to allow in regression
    MaxIntOrder = 2#d  for high dimentional problems it should be fixed at 2

    # Maximum number of function evaluation
    MaxNumFunEval = 100 

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
 
       ' Set value of Pd ' # Already set
       ' Generate initial design of size n=n0 from scrambled Sobol sequence.' 
       #Low discrepancy quasi-random sequences, e.g. Sobol sequences, fill a space more uniformly than uniformly random sequences.
       ' Set k=0. '
    """

    #Generate initial design (Dataset)
    n = n0
    #Matlab code : QuasiMC = sobolset(d)
    #              QuasiMC=scramble(QuasiMC,'MatousekAffineOwen')
    #              D=net(QuasiMC,n)*2-1
    D = scrambled_sobol(d, n) * 2 - 1 # returns the first n points from the Sobol sequence
    # remeber, If G_i is the CDF of u_i, then v_i=2 * G(u_i) -1 is uniformly distributed in [-1, 1]. 


    #QuasiMCResampling = scrambled_sobol(d, n) #Resampling option (Not used in this version)
    #ReSampleIndex = 1


    """
     2. Run experiment: evaluate function at all points in design. i.e. find Y = f(X); for all samples in D
 
    """
    

    Y = MyFun(D,lb,ub,ObjInd,ProblemName,k) # ProblemName must be P_objective or P_objective2

    # Example 1. p159  - Ishigami function (test problem)
    #Y = np.sin(np.pi*(D[:,0]+1)-np.pi) + 7*(np.sin(np.pi*(D[:,1]+1)-np.pi))**2 + 0.1*(np.pi*(D[:,2]+1)-np.pi)**4 * np.sin(np.pi*(D[:,0]+1)-np.pi)

    """
     3.1. Generate regressors
    """
    # Generate regressors
    (X, alpha, AnovaIndicators, Lambda) = MultivariateLegendre(D, Pd, MaxIntOrder)
    Nf = AnovaIndicators.shape[0]
    M = X.shape[1]
    M2 = M-1

    #Interaction order and number of levels for each component of delta
 
    E = np.array([sum(row) for row in AnovaIndicators]).T
    F = Pd - E + 2

    #Number of Sobol indices to compute
    NI = sum(E <= truncate)

    """
    3.2. Starting models for global search
    """
    if d > 1 and Pd > 5:   # eq. 27 p 157
        NoStart0 = 2 ** d - 1 + 2 ** d -1 - d
        deltastart0 = np.zeros((NoStart0, Nf))
        deltastart0[0:d,0:d] = Pd * np.identity(d)
        No = d
        for i in range(2,d+1):
            Factors = np.array(list(itertools.combinations(range(1,d+1), i))) # This matrix provide different order of rows for some indices compared to the Matlab code -> check if it needs to be the same as the Matlab code
            if i % 2 == 0:  # ** It works only for d=4 (provide the same order)
                Factors = Factors[::-1,:] # reverse sort the array ** It works only for d=4
            NoFactors = Factors.shape[0]
            for j in range(NoFactors):
                Include = np.logical_and(((E - np.array([sum(row) for row in AnovaIndicators[:,Factors[j,:]-1]])) == 0), (E <= 2))
                No += 1
                deltastart0[No-1,Include] = Pd
                No += 1
                deltastart0[No-1,Include] = 2

    elif d > 1 and Pd <= 5:  # eq. 27 and 28 - p 157-158
        NoStart0 = 2 ** d - 1 
        deltastart0 = np.zeros((NoStart0, Nf))
        deltastart0[0:d,0:d] = Pd * np.identity(d)
        No = d
        for i in range(2,d+1):
            Factors = np.array(list(itertools.combinations(range(1,d+1), i)))
            #Factors = Factors[::-1,:] # reverse sort the array 
            NoFactors = Factors.shape[0]
            for j in range(NoFactors):
                Include = np.logical_and(((E-np.array([sum(row) for row in AnovaIndicators[:,Factors[j,:]-1]]).T) == 0), (E <= 2))
                No += 1
                deltastart0[No-1,Include] = Pd

    elif d == 1:
        deltastart0 = np.arange(Pd+1)[np.newaxis].T
        NoStart0 = Pd+1



    """
    Initialize some variables
    """
    iteration = 0
    stop = 0
    StoremdMinusEta = np.zeros((30,M))# Generate an empty 2D 30*M list of lists, 30 is the max number of iteration, check if need to increase it
    BestModel = []  # check if need to use different definition such as 'a *[None]'
    PredCriterion1 = [2 * [None] for i in range(30)] # Generate an empty 2D 30*2 list of lists, 30 is the max number of iteration, check if need to increase it
    PredCriterion2 = [2 * [None] for i in range(30)] # Generate an empty 2D 30*2 list of lists, 30 is the max number of iteration, check if need to increase it
   
    while stop == 0:
    
        if iteration > 0: 
            # Add the best model found in the previous iteration to the set of starting models
            if  (cdist(BestModel, deltastart0)[iteration-1]).min() > 0.5:  # check if need to use min(a) intead
                deltastart = np.vstack((deltastart0, BestModel[iteration-1]))
                NoStart = NoStart0 + 1
        else:
            deltastart = deltastart0
            NoStart = NoStart0
            
        if n > n0:
            XAdd = MultivariateLegendre2(D[(n-nadd):n,:],Pd,MaxIntOrder)
            X = np.vstack((X, XAdd))
        
        
        #Compute the centered response used in model selection
        mY = Y.mean(0) # Y must be array or matrix (np.array or np.matrix)
        Y2 = Y - X[:,0] * mY
    
        #Precompute some matrices to speed up computation of Q(delta)
    
        W = X[:,1:M]
        W2W2T0 = gamma0 * np.matrix(X)[:,0] * np.matrix(X)[:,0].T    
    
        # To avoid changing in X in the following for while W is modifying (in an objective oriented language such as Python) we need to define a temporary variable Wtemp
        Wtemp = np.zeros(np.shape(W))
        for i in range(M2):
            Wtemp[:,i] = W[:,i] * np.sqrt(gamma * r ** (Lambda[1,i]-1))
    
        W = Wtemp # Now we refer W to Wtemp
    
        R00 = W2W2T0 + np.identity(n)    
        
        """
        4. Global Search - p157-159
    
        Modiﬁed Fedorov algorithm is utilized instead of the Fedorov algorithm (Cook and Nachtsheim, 1980) employed by Tan and Wu (2013).
        """
        deltaopt = np.zeros((NoStart,Nf))
        Q = np.zeros((NoStart,1))
        print('Total number of starting points is ', NoStart, '.')
    
        for i in range(NoStart):
            [deltaopt[i,:], Q[i]] = Search(deltastart[i,:], Lambda, Nf, W, Y2, d, n, Pd, F, lrho, E, R00) 
            #print('Objective =', ObjInd, '. Function evaluation = ', n, '. Iteration =', iteration,'. Global search from ', i, '/', NoStart,' starting points completed.')
       
        iteration += 1
        [MinQ, IndexMinQ] = Q.min(), Q.argmin() 
        BestModel.append(deltaopt[IndexMinQ,:])
    
        print('Best model found')
        print(np.array(BestModel)[iteration-1,:])   
    
        #Precompute quantities needed in computing credible limits
        delta = np.array(BestModel)[iteration-1,:]
        check = Lambda[1,:] <= delta[Lambda[0,:].astype(int)] # check if (Lambda[1,:]-1) <= delta[Lambda[0,:].astype(int)]
        check2 = np.where(check)[0]
        check3 = np.hstack((0, check2 + 1)) 
        R = R00 + np.matmul(np.matrix(W)[:,check2], np.matrix(W)[:,check2].T)
        [IR, LDR] = invandlogdet(R)
        S = np.hstack((gamma0, gamma*(r**(Lambda[1,:]-1))))
        S = np.diag(S[check3])
        XS = np.matrix(X)[:,check3] @ np.matrix(S)
        Gd = np.matrix(S) - XS.T @ IR @ XS
        RSS = np.matmul(np.matmul((Y2.T),IR), Y2)
        Sigma2Est = RSS / n 
        md = np.vstack((mY, np.zeros((len(check2),1)))) + np.matmul(np.matmul((XS.T),IR), Y2).T #XS.T * IR * Y2  # Needs to be check after calculating the Y (mY may need to be an scalar)
        StoremdMinusEta[iteration-1, check3] = (md - np.vstack((mY, np.zeros((len(check2),1))))).T  # if got error change check3 to np.matrix(check3)
    
        weights = 1 / np.diag(IR)
     
        # The ratio of the RMS of Y1 −X1m(K, −1), . . . , Yn −Xnm(K, −n) and the standard deviation of Y1, . . . , Yn (Criterion 1) is computed.
        # It is percentage then must be <= 100 --> check the formula below
        PredCriterion1[iteration-1][0] = np.sqrt((np.square(weights * (np.squeeze(np.asarray(np.matrix(Y).T - (X[:,check3] @ md)))))).mean(0)) / np.std(Y,0) * 100   
  
        # Numpy matrix to array: A = np.squeeze(np.asarray(M))
        print('RMS of leave-one-out prediction errors/Standard deviation of Y (in percent)')
        print(PredCriterion1[iteration-1][0])

        """
        5. Check if stopping conditions have been met 
            5.1. θ(h_1) < tolerance 3;
            5.2. Criterion 1 at iteration K is less than tolerance 1;
            5.3. Criterion 2 at iteration K − 1 is less than tolerance 2
        
        Also compute θ(h1) = 100 max{||m(K−h1+1)−m(K−h1+2)||_2, . . . , ||m(K−1) − m(K)||_2} / ||m(K) − η||_2 
        to determine whether the changes in the estimate of β have become suﬃciently small over the most +
        recent h1 iterations.    
    
    
        The sequential addition of design points is also terminated with the best model δK if
        θ(h2) < tolerance 3; i.e., the changes in the estimate of β over h2 consecutive iterations is
        judged to be small relative to the change from prior to posterior mean of β at iteration K,
        where h2 > h1.
        """
    

        if iteration >= h1: # check the right index of iteration and if there is a need to reduce it by 1 unit (i.e. replace 'iteration' by 'iteration-1')
            CheckPredAccuracy = np.logical_and((PredCriterion1[iteration-1][0] < Tolerance1), (PredCriterion2[iteration-2][0] < Tolerance2))  # Steps 5.2 & 5.3 in the flowchart p155
            PercentChangeInBetaEst = np.sqrt((((StoremdMinusEta[(iteration-h1):(iteration-1),] - StoremdMinusEta[(iteration-h1+1):iteration,]) ** 2).sum(1)).max(0)) / np.sqrt((StoremdMinusEta[iteration,:] ** 2).sum(0)) * 100
            CheckPercentChangeInBetaEst = PercentChangeInBetaEst < Tolerance3  # Step 5.1 ( check if θ(h1) < tolerance 3)
            if CheckPredAccuracy and CheckPercentChangeInBetaEst:
                stop=1
                print('Procedure terminated. Changes in posterior mean of regression coefficients are small and prediction criteria are less than tolerances.')
        
  
        if (stop==0) and (iteration >= h2):
            PercentChangeInBetaEst = np.sqrt((((StoremdMinusEta[(iteration-h2):(iteration-1),] - StoremdMinusEta[(iteration-h2+1):iteration,]) ** 2).sum(1)).max(0)) / np.sqrt((StoremdMinusEta[iteration,:] ** 2).sum(0)) * 100
            CheckPercentChangeInBetaEst = PercentChangeInBetaEst < Tolerance3
            if CheckPercentChangeInBetaEst:
                stop=1
                print('Procedure terminated. Changes in posterior mean of regression coefficients are small but prediction criterion/criteria does/do not meet tolerance(s).')
        
 
        if iteration >= 4: #30:  # we can have a termination commond for max iteration  (particularly for high dimentional problems)
            stop=1
            print('Maximum iteration is reached')
    
        if n >= MaxNumFunEval:
            stop=1
            print('Maximum number of function evaluations is reached')
    
      
        if stop==0:
        
            # Next design point is set to the next nadd Sobol quasi random points
            NextDesignPoints = scrambled_sobol(d, n+nadd)
            NextDesignPoints = NextDesignPoints[n:(n+nadd),] * 2 - 1  
        
        
            """
            Step 7.
            If the two sets of stopping criteria above are not satisﬁed, then the next nA
            points un+1, . . . , un+nA from the scrambled Sobol sequence are added to the design. The HPP
            model is used to predict the output at the additional points, and the true function is evaluated
            at these points. The RMS of Yn+1 − xn+1m(K), . . . , Yn+nA − xn+nAm(K) normalized by the
            standard deviation of Y1, . . . , Yn+nA (Criterion 2) is computed. 
            Finally, the value of n is then updated to n + nA, 
            and the steps stated in this paragraph are repeated.
  
            """
            #Active this for final version        
            # Evaluate true function at new design points  
            NewObservations = MyFun(NextDesignPoints,lb,ub,ObjInd,ProblemName,k)
        
             # This is just for test
            # Deactive this for final version
            # Example 1. p159  - Ishigami function
            #NewObservations = np.sin(np.pi*(NextDesignPoints[:,0]+1)-np.pi) + 7*(np.sin(np.pi*(NextDesignPoints[:,1]+1)-np.pi))**2 + 0.1 * (np.pi * (NextDesignPoints[:,2]+1)-np.pi)**4 * np.sin(np.pi*(NextDesignPoints[:,0]+1)-np.pi)

        
            # Compute predictions for new design points
            predictions = Pred(NextDesignPoints, md, check3, Pd, MaxIntOrder)
        
            # Update design and vector of function evaluations
            D = np.vstack((D, NextDesignPoints))
            Y = np.concatenate((Y, NewObservations))
       
            #Increase n by nadd
            n += nadd
        
            # Evaluate standard deviation of prediction errors and
            RMSPredErr = np.sqrt(np.square(np.matrix(NewObservations).T - predictions).mean(0))
            print('RMS of prediction errors/Standard deviation of Y (in percent)')  # check the values
            PredCriterion2[iteration-1][0] = RMSPredErr / np.std(Y,0) * 100 # check the index of iteration
            print(PredCriterion2[iteration-1][0])
        
    # end of while loop

    """
    Results are store as the follows

    Need to set a right shape, e.g. see if better to use dictionary
    """
    # print('Value of P')
    # print(P)
    # print('Initial design size')
    # print(n0)
    # print('Final design size')
    # print(n)

    #StoreData.D = D # Final design
    #StoreData.Y = Y # observations

    # print('Final design and observations')
    # print(StoreData)

    #StoreBestModelsIteration = np.hstack(( np.arange(0,iteration-1).T, BestModel))

    # print('Iteration number, and best model for each iteration')
    # print(StoreBestModels)
    # print('ANOVA decomposition index (maps each component of delta to the functional ANOVA component it represents)')
    # print(AnovaIndicators)

    #StorePredictionInfo = np.hstack(( np.arange(0,iteration-1).T, PredCriterion1, np.vstack((PredCriterion2, -1)) ))
    # print('Iteration number, prediction criterion 1, prediction criterion 2')
    # print(StorePredictionInfo)

    # xlswrite(Filename,{'d, initial design size, final design size, P, maximum order of interactions, tolerance 3, tolerance 1, tolerance 2, rho, gamma0, gamma, r'},'1','A1')
    # xlswrite(Filename,[d n0 n Pd MaxIntOrder Tolerance3 Tolerance1 Tolerance2 rho gamma0 gamma r],'1','A2')
    # 
    # xlswrite(Filename,{'Set of starting models'},'M_0','A1')
    # xlswrite(Filename,deltastart0,'M_0','A2')
    # 
    # xlswrite(Filename,{'Final design and observations'},'1','A4')
    # xlswrite(Filename,StoreData,'1','A5')

    # xlswrite(Filename,{'Iteration number and best model for each iteration'},'2','A1')
    # xlswrite(Filename,StoreBestModels,'2','A2')
    # xlswrite(Filename,{'ANOVA decomposition index (maps each component of delta to the functional ANOVA component it represents)'},'2',['A' num2str(iteration+3)])
    # xlswrite(Filename,AnovaIndicators*1,'2',['A' num2str(iteration+4)])
    # 
    # xlswrite(Filename,{'Iteration number, prediction criterion 1, prediction criterion 2'},'3','A1')
    # xlswrite(Filename,StorePredictionInfo,'3','A2')                
                     
    Data.md = md  # check if there is need to use different type of srorage for the following data
    Data.check3 = check3
    Data.Pd = Pd
    Data.MaxIntOrder = MaxIntOrder
    Data.D = D
    Data.Y = Y
    Data.iteration = iteration

    if d > 1:
        TotalIndices = SimulateSobolIndices((np.array(BestModel)[iteration-1,:]).astype(int), X, W, Y2, mY, d, n, M2, Nf, R00, gamma0, gamma, r, AnovaIndicators, Lambda, NI)
        Data.TotalIndices = TotalIndices
        
   
    return Data    #(md, check3, P, MaxIntOrder, D, Y, iteration, TotalIndices) # check what else we need to return


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    