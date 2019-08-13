# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:52:00 2019

@author: babshava
"""

"""
Some user defined functions
"""
import numpy as np
import pandas as pd
from itertools import repeat
from scipy.optimize import fsolve
import math
from scipy import stats
#from MultivariateLegendre import orthonormal_polynomial_legendre, MultivariateLegendre2, MultivariateLegendre

def Pred(x0, md, check3, P, MaxIntOrder):
    """
    The HPP model is used to predict the output.
    
    Get md, check3, P and MaxIntOrder instead of using them as the global variables
    #global md check3 P MaxIntOrder
    """
    
    x0 = MultivariateLegendre2(x0,P,MaxIntOrder)
    x0 = x0[:,check3]
    return (x0 * md) # the product '*' here probably needs to replace with other products e.g. np.matmul(x0, md)


def Search(delta, Lambda, Nf, W, Y2, d, n, P, F, lrho, E, R00): # check if need to send Nf to the function
    """
    describtion
    """
    #global Nf
    localoptimum = 0
    while localoptimum ==0:
        olddelta = delta
        [oldm2lprob, IR, LDR] = calculatem2lprob2(olddelta, Lambda, W, Y2, d, n, P, Nf, F, lrho, E, R00)
        for i in range(Nf):
            [m2lprob, values, storeIR, storeLDR] = calculatem2lprob(delta,i,oldm2lprob,IR,LDR, W, Y2, d, n, P, Nf, F, lrho, E, R00, Lambda)
            [Minm2lprob, index] = m2lprob.min(), m2lprob.argmin() # check if need to use min(m2lprob) instead
            delta[i] = values[index]
            oldm2lprob = Minm2lprob
            IR = storeIR[index]
            LDR = storeLDR[index]
        
        if (sum(olddelta==delta) == Nf).all():
            localoptimum = 1
        
        
    Model = delta
    Q  = oldm2lprob
    
    return (Model, Q)
    

def calculatem2lprob(delta0,I,m2lprob0,IR,LDR, W, Y2, d, n, P, Nf, F, lrho, E, R00, Lambda): # Needs to be checked for errors
    """
    describtion:
        
    """

    #global W, Y2, d, n, P, Nf, F, lrho
    #global E, Lambda
    
    delta = delta0  
    values = [0] + list(range(E[I], P+1))
    storeIR = [None] * F[I]
    storeLDR = np.zeros((F[I]))
    calculatem2lprob = math.inf * np.ones(F[I])
    index = np.where(abs(values - delta[I])<0.5)
    calculatem2lprob[int(index[0])] = m2lprob0
    storeIR[int(index[0])] = IR
    storeLDR[int(index[0])] = LDR
          
    check0 = Lambda[1,:] <= delta[Lambda[0,:].astype(int)]
    check20 = np.where(check0)[0]
    logpdelta0 = sum(delta[0:d] * lrho) + sum((np.maximum(delta[d:Nf] - E[d:Nf] + 1, 0)) * E[d:Nf] * lrho)

    check2 = check20
    logpdelta=logpdelta0
    
    for k in range(int(index[0])+1,F[I]):
        oldcheck = check2
        delta[I] = values[k]
        check = Lambda[1,:] <= delta[Lambda[0,:].astype(int)]
        check2 = np.where(check)[0]
        newcheck = np.setdiff1d(check2,oldcheck)
        L = len(newcheck)
        T = np.identity(L) + np.matmul(np.matmul((np.matrix(W)[:,newcheck].T),IR), np.matrix(W)[:,newcheck])  #np.matrix(W)[:,newcheck].T * IR * np.matrix(W)[:,newcheck]
        T2 = np.matmul(np.matrix(W)[:,newcheck].T, IR)
        [IT, LDT] = invandlogdet(T)
        IR = IR - np.matmul(np.matmul((T2.T),IT), T2) # T2.T * IT * T2
        LDR = LDT + LDR
        storeIR[k] = IR
        storeLDR[k] = LDR
        RSS = np.matmul(np.matmul((Y2.T),IR), Y2) 
        logpdelta = logpdelta + E[I] * lrho
        m2lprob = n * np.log(RSS) + LDR - 2 * logpdelta
        calculatem2lprob[k] = m2lprob  # Check if it is correct


    check2 = check20
    logpdelta = logpdelta0
    IR = storeIR[int(index[0])]
    LDR = storeLDR[int(index[0])]
    
    for k in range(int(index[0])-1,-1,-1):
        oldcheck = check2
        delta[I] = values[k]
        check = Lambda[1,:] <= delta[Lambda[0,:].astype(int)]
        check2 = np.where(check)[0]
        newcheck = np.setdiff1d(oldcheck,check2)
        L = len(newcheck)
        T = np.identity(L) -  np.matmul(np.matmul((np.matrix(W)[:,newcheck].T),IR), np.matrix(W)[:,newcheck])
        T2 = np.matmul(np.matrix(W)[:,newcheck].T, IR) #np.matrix(W)[:,newcheck].T * IR
        [IT, LDT] = invandlogdet(T)
        IR = IR + np.matmul(np.matmul((T2.T),IT), T2) #T2.T * IT * T2
        LDR = LDT + LDR
        storeIR[k] = IR
        storeLDR[k] = LDR
        RSS = np.matmul(np.matmul((Y2.T),IR), Y2)  
        logpdelta = logpdelta - E[I] * lrho
        m2lprob = n * np.log(RSS) + LDR - 2 * logpdelta
        calculatem2lprob[k] = m2lprob  # Check if it is correct 
  
    return (calculatem2lprob, values, storeIR, storeLDR)
    
    
def calculatem2lprob2(delta, Lambda, W, Y2, d, n, P, Nf, F, lrho, E, R00): # Checked --> OK
    """
    describtion
    """

    #global W, Y2, d, n, Nf, R00, lrho
    #global E, Lambda
    
    check = Lambda[1,:] <= delta[Lambda[0,:].astype(int)]
    check2 = np.where(check)[0]
    R = R00 + np.matrix(W)[:,check2] * np.matrix(W)[:,check2].T
    [IR, LDR] = invandlogdet(R)
    storeIR = IR
    storeLDR = LDR
    RSS = np.matmul(np.matmul((Y2.T),IR), Y2) 
    logpdelta = sum(delta[0:d] * lrho) + sum((np.maximum(delta[d:Nf] - E[d:Nf] + 1, 0)) * E[d:Nf] * lrho)
    calculatem2lprob = n * np.log(RSS) + LDR - 2 * logpdelta
    
    return (calculatem2lprob, storeIR, storeLDR)

    
    
def invandlogdet(R):
    """
    Takes a positive definite matrix R
    
    Returns the inverse and log determinant of R from it's Cholesky decomposition.
    
    Since (I + X S_δ X^T) is positive deﬁnite, both its inverse((I + X S_δ X^T)^{-1}) and log determinant(ln(|I + X S_δ X^T|)) can be obtained from its Cholesky
    decomposition.
    
    
    """

    CR = np.linalg.cholesky(R).T # Cholesky factorization
    ICR = np.linalg.inv(CR) # Build the Inverse 
    invR = np.matrix(ICR) * np.matrix(ICR).T
    logdetR = 2 * (np.log(np.diag(CR))).sum(axis=0)
    
    return (invR, logdetR)
    



def CheckIfPosDef(A):
    """
    Check if a matrix is positive definite
      1. Check if the matrix is symmetric
      2. Check if all the eigenvalues of matrix are positive
      
     return True if the matrix is positive definite and return False if not
    """
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def SimulateSobolIndices(model, X, W, Y2, mY, d, n, M2, Nf, R00, gamma0, gamma, r, AnovaIndicators, Lambda, NI):
    """
    Describtion:
    """
    #global X, W, Y2, mY, d, n, M2, Nf, R00, gamma0, gamma, r
    #global AnovaIndicators, Lambda
    global SobolIndices, TotalSensitivityIndices, I, Type, quantile
    #global Filename, NI
        
    SampleSize0 = 100
    delta = model
    check = Lambda[1,:] <= delta[Lambda[0,:].astype(int)]  # check if (Lambda[1,:]-1) <= delta[Lambda[0,:].astype(int)] # For test and compare with matlab code check = np.hstack((np.ones((5)), np.zeros((80)), np.ones((40)), np.zeros((40)), np.ones((80)), np.zeros((5)), 1 ))
    check2 = np.where(check)[0]  
    check3 = np.hstack((0, check2 + 1))  
    L = len(check3)
        
    if L==1:
        #print('ANOVA Decomposition Index (columns 1:d), Probability=0, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile) for Sobol Indices')
        Sobol = np.hstack((AnovaIndicators[0:NI,:], np.ones((NI,1)), np.zeros((NI,3))))
        # print(Sobol)
        # xlswrite(Filename,{'ANOVA Decomposition Index (columns 1:d), Probability=0, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile) for Sobol Indices'},'SobolIndices','A1')
        # xlswrite(Filename,Sobol,'SobolIndices','A2')
        # print('Probability=0, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile) for Total Sensitivity Indices')
        Total = np.hstack((np.ones((d,1)), np.zeros((d,3))))
        #disp(Total)
        #xlswrite(Filename,{'Probability=0, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile) for Total Sensitivity Indices'},'TotalSensitivityIndices','A1')
        #xlswrite(Filename,Total,'TotalSensitivityIndices','A2')

    R = R00 + np.matmul(np.matrix(W)[:,check2], np.matrix(W)[:,check2].T)
    [IR, LDR] = invandlogdet(R)
    S = np.hstack((gamma0, gamma*(r**(Lambda[1,:]-1))))
    S = np.diag(S[check3])
    XS = np.matrix(X)[:,check3] @ np.matrix(S)
    Gd = np.matrix(S) - XS.T @ IR @ XS # XS.T * IR * XS
    RSS = np.matmul(np.matmul((Y2.T),IR), Y2) 
    scale = Gd * np.matrix(RSS / n)[0,0] # 
    md = np.vstack((mY, np.zeros((L-1,1)))) + (XS.T @ IR @ Y2).T #XS.T * IR * Y2  
    
    
    if CheckIfPosDef(scale): #First check if scale is positive definite
        cholscaleT = np.linalg.cholesky(scale) # Cholesky factorization
    else:
        eigvec, eigval, VT = np.linalg.svd(scale)
        cholscaleT = np.multiply(eigvec, np.sqrt(eigval)) @ eigvec.T 
        
    OldSampleSize = 0
    NewSampleSize = SampleSize0
    stop = 0
    
    SobolIndices = [NewSampleSize * [None] for i in range(NI)] # Generate a 2D empty NI*NI list of lists # check if we need to use NewSampleSize instead of NI
    TotalSensitivityIndices = [NewSampleSize * [None] for i in range(d)] # check if we need to use NewSampleSize instead of NI
    coeffind = NI * [None]  # Use NI * [None] to generate an empty list
    coeffind2 = d * [None] 
    
    for j in range(NI):
        coeffind[j] = Lambda[0,:] == j
    
    for j in range(d):
        Indicator = AnovaIndicators[:,j]
        coeffind2[j] = np.zeros((1,M2))
        for i in range(Nf):
            if Indicator[i] > 0:
                coeffind2[j] = (coeffind2[j] + (Lambda[0,:] == i)) > 0
    
        
    while stop == 0:
        for k in range(int(OldSampleSize), int(NewSampleSize)):
            beta = np.zeros((M2,1))
            sample = np.matmul(cholscaleT, np.random.normal(0, 1, size=(L,1))) / np.sqrt(stats.chi2.rvs(n)/n) + md  
            #np.random.normal(mu, sigma, size=(m,n)) generate an m*n matrix includes samples from the Normal distribution with mean mu and standard deviation sigma
            #R = stats.chi2.rvs(V) generates random numbers from the chi-square distribution with degrees of freedom parameters specified by V. The degrees of freedom parameters in V must be positive.
            beta[check2,0] = sample[1:L].T
            SSbeta = (beta**2).sum()
            for j in range(NI): 
                SobolIndices[j][k] = (beta[coeffind[j]]**2).sum() / SSbeta
                
            for j in range(d):
                TotalSensitivityIndices[j][k] = (beta[coeffind2[j][0]]**2).sum() / SSbeta
                
        Est = np.zeros((NI,1))
        VarianceEst = np.zeros((NI,1))
        Variance = np.zeros((NI,1))
        
        for j in range(NI):
            Est[j] = np.mean(SobolIndices[j])
            Variance[j] = np.var(SobolIndices[j]) * NewSampleSize/(NewSampleSize-1)
            VarianceEst[j] = Variance[j] / NewSampleSize
        
        Stderr = np.sqrt(VarianceEst)
    
        Est2 = np.zeros((d,1))
        VarianceEst2 = np.zeros((d,1))
        Variance2 = np.zeros((d,1))
        
        for j in range(d):
            Est2[j] = np.mean(TotalSensitivityIndices[j])
            Variance2[j] = np.var(TotalSensitivityIndices[j]) * NewSampleSize/(NewSampleSize-1)
            VarianceEst2[j] = Variance2[j] / NewSampleSize
        
        Stderr2 = np.sqrt(VarianceEst2)
    
        if np.all(Stderr <= 0.00025) and np.all(Stderr2 <= 0.00025):
            stop=1
        else:
            TotalSampleSizeNeeded = (np.vstack((Variance, Variance2)) / 0.00025 ** 2).max(0)
            OldSampleSize = NewSampleSize
            NewSampleSize = np.ceil(TotalSampleSizeNeeded)
        
    # End of while 
    
    LCL = np.zeros((NI,1))
    UCL = np.zeros((NI,1))
    Pr0 = np.zeros((NI,1))
    Pr1 = np.zeros((NI,1))
    STD = np.sqrt(Variance)
    Type = 1         
        
    for I in range(NI):
        Pr0[I] = CDFEst(5 * 10 ** (-4), SobolIndices, TotalSensitivityIndices, Type, I)
        Pr1[I] = 1 - CDFEst(1 - 5 * 10 ** (-4), SobolIndices, TotalSensitivityIndices, Type, I)
        if STD[I] >= (1.25 * 10 ** (-4)):
            if Pr0[I] > 0.025:
                LCL[I] = 0
            else:
                quantile = 0.025
                LCL[I] = fsolve(CdfDev, max(Est[I] - STD[I], 0), quantile)  #  e.g. of a function: f = lambda x : x * np.cos(x-4)
            
            if Pr1[I] > 0.025:
                UCL[I] = 1
            else:
                quantile = 0.975
                UCL[I] = fsolve(CdfDev, min(Est[I] + STD[I], 1), quantile)
             
        else:
            LCL[I] = Est[I]
            UCL[I] = Est[I]
        
        
            
    LCL2 = np.zeros((d,1))
    UCL2 = np.zeros((d,1))
    Pr0_2 = np.zeros((d,1))
    Pr1_2 = np.zeros((d,1))
    STD2 = np.sqrt(Variance2)
    Type = 2      
      
    for I in range(d):
        Pr0_2[I] = CDFEst(5 * 10 ** (-4), SobolIndices, TotalSensitivityIndices, Type, I)
        Pr1_2[I] = 1 - CDFEst(1 - 5 * 10 ** (-4), SobolIndices, TotalSensitivityIndices, Type, I)
        if STD2[I] >= (1.25 * 10 ** (-4)):
            if Pr0_2[I] > 0.025:
                LCL2[I] = 0
            else:
                quantile = 0.025
                LCL2[I] = fsolve(CdfDev, max(Est2[I] - STD2[I], 0), quantile) 
            
            if Pr1_2[I] > 0.025:
                UCL2[I] = 1
            else:
                quantile = 0.975
                UCL2[I] = fsolve(CdfDev, min(Est2[I] + STD2[I], 1), quantile)
            
        else:
            LCL2[I] = Est2[I]
            UCL2[I] = Est2[I]
        
      
            
    # display('ANOVA decomposition index (columns 1:d), Probability<=0.0005, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile), Probability>=0.9995 for Sobol Indices')
    # display('All values (except the ANOVA decomposition index, which are binary numbers) are rounded to three decimal places')
    Sobol = np.hstack((AnovaIndicators[0:NI,:], np.round(np.hstack([Pr0, LCL, Est, UCL, Pr1]).dot(10 **3)) / 10**3 ))  
    # disp(Sobol)                               
    # xlswrite(Filename,{'ANOVA decomposition index (columns 1:d), Probability<=0.0005, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile), Probability>=0.9995 for Sobol Indices'},'SobolIndices','A1')
    # xlswrite(Filename,{'All values (except the ANOVA decomposition indices, which are binary numbers) are rounded to three decimal places'},'SobolIndices','A2')
    # xlswrite(Filename,Sobol,'SobolIndices','A3')

    # display('Probability<=0.0005, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile), Probability>=0.9995 for Total Sensitivity Indices')
    # display('All values are rounded to three decimal places')
    
    Total = np.round(np.hstack([Pr0_2, LCL2, Est2, UCL2, Pr1_2]).dot(10 **3)) / 10**3
    
    # disp(Total)
    TotalIndices = Total[:,1:-1]  
    # xlswrite(Filename,{'Probability<=0.0005, LCL (0.025 Quantile), Mean, UCL (0.975 Quantile), Probability>=0.9995 for Total Sensitivity Indices'},'TotalSensitivityIndices','A1')
    # xlswrite(Filename,{'All values are rounded to three decimal places'},'TotalSensitivityIndices','A2')
    # xlswrite(Filename,Total,'TotalSensitivityIndices','A3')
    
    return TotalIndices
    
    
def CdfDev(x, quantile):
    """
    describtion
    """
    #global quantile
    
    CdfDev = CDFEst(x, SobolIndices, TotalSensitivityIndices, Type, I) - quantile
    return CdfDev
    
    
def CDFEst(x, SobolIndices, TotalSensitivityIndices, Type, I):
    """
    describtion
    """
    #global SobolIndices, TotalSensitivityIndices, Type, I
    
    if x < 0:
        CDFEst = 0
    elif x >= 1:
        CDFEst = 1
        
    if Type == 1:
        CDFEst = PiecewiseLinearCDF(x,SobolIndices[I]) # needs test for real problem
    elif Type == 2:
        CDFEst = PiecewiseLinearCDF(x,TotalSensitivityIndices[I]) # needs test for real problem
    
    return CDFEst
    
    
def PiecewiseLinearCDF(x,data):
    """
    describtion
    """
    [Fi,xi] = ecdf(data)
    nxi = len(xi)
    if nxi <= 1:
        PiecewiseLinearCDF = xi[0] <= x
    else:
        nxi -= 1
        xj = xi
        Fj = (np.hstack((0,Fi[0:-1])) + Fi) / 2
        MinPoint = max(xj[0] - Fj[0] * (xj[1] - xj[0]) / (Fj[1] - Fj[0]), 0)
        MaxPoint = min(xj[nxi] + (1 - Fj[nxi]) * ((xj[nxi] - xj[nxi-1]) / (Fj[nxi] - Fj[nxi-1])), 1)
        xj = np.hstack((MinPoint, xj, MaxPoint))
        Fj = np.hstack((0, Fj, 1))
        index = sum(xj <= x)
        if index==0:
            PiecewiseLinearCDF = 0
        elif index==(nxi + 3):
            PiecewiseLinearCDF = 1
        else:
            PiecewiseLinearCDF = Fj[index-1] + (Fj[index] - Fj[index-1]) / (xj[index] - xj[index-1]) * (x - xj[index-1]) 
    
    return PiecewiseLinearCDF
    
def ecdf(sample):
    """
        [f,x] = ecdf(y) returns the empirical cumulative distribution function (cdf), f, evaluated at the points in x, using the data in the vector y.

        In survival and reliability analysis, this empirical cdf is called the Kaplan-Meier estimate. And the data might correspond to survival or failure times.
    """
    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return  cumprob, quantiles    
    
    