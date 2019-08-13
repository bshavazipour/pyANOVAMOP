# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:52:00 2019

@author: babshava
"""

"""
Some user defined functions
"""

import pandas as pd

def Pred(x0,md, check3, P, MaxIntOrder):
    """
    Get md, check3, P and MaxIntOrder instead of using them as the global variables
    #global md check3 P MaxIntOrder
    """
    
    x0 = MultivariateLegendre2(x0,P,MaxIntOrder)
    x0 = x0[:,check3]
    return (x0 * md)


def Search(delta, Nf): # Needs to be checked for errors
    """
    describtion
    """

    #global Nf
    localoptimum = 0
    while localoptimum ==0:
        olddelta = delta
        [oldm2lprob, IR, LDR] = calculatem2lprob2(olddelta)
        for i in range(Nf)
            [m2lprob, values, storeIR, storeLDR] = calculatem2lprob(delta,i,oldm2lprob,IR,LDR)
          ¤  [Minm2lprob, index] = min(m2lprob)
            delta[i] = values[index]
            oldm2lprob = Minm2lprob
            IR = storeIR[index]
            LDR = storeLDR[index]
        
    ¤    if sum(olddelta==delta) == Nf
            localoptimum = 1
        
        
    Model = delta
    Q  = oldm2lprob
    
    return (Model, Q)
    

def calculatem2lprob(delta0,I,m2lprob0,IR,LDR): # Needs to be checked for errors
    """
    describtion:
        
    """

    global W Y2 d n P Nf F  lrho
    global E Lambda
    
    delta = delta0  # delta0 probably is deltastart0
    values = [0] + list(range(E[I], P+1))
    storeIR = [None] * F[I]
    storeLDR = np.zeros((F[I]))
    calculatem2lprob = math.inf * np.ones(F[I])
    index = np.where(abs(values - delta[0][I])<0.5)
    calculatem2lprob[int(index[0])] = m2lprob0
    storeIR[int(index[0])] = IR
    storeLDR[int(index[0])] = LDR
    
    
    
    ¤check0 = Lambda[1,:] <= delta(Lambda[0,:])
    check20 = np.where(check0)
    logpdelta0 = sum(delta[0][:d] * lrho) + sum((np.maximum(delta[0][d:Nf] - E[d:Nf] + 1, 0)) * E[d:Nf] * lrho)

    check2 = check20
    logpdelta=logpdelta0
    
    for k in range(int(index[0])+1,F[I]):
        oldcheck = check2
        delta[I] = values[k]
        check = Lambda[1,:] <= delta(Lambda[0,:])
        check2 = np.where(check)
        newcheck = np.setdiff1d(check2,oldcheck)
        L = len(newcheck)
        T = np.identity(L) + np.matrix(W)[:,newcheck].T * IR * np.matrix(W)[:,newcheck]
        T2 = np.matrix(W)[:,newcheck].T * IR
        [IT, LDT] = invandlogdet(T)
        IR = IR - T2.T * IT * T2
        LDR = LDT + LDR
        storeIR[k] = IR
        storeLDR[k] = LDR
        RSS = (Y2.T) * IR * Y2 # Y2 must be a matrix; otherwise use np.matrix(Y2) 
        logpdelta = logpdelta + E(I) * lrho
        m2lprob = n * np.log(RSS) + LDR - 2 * logpdelta
        calculatem2lprob[k] = m2lprob  # Check if it is correct


    check2 = check20
    logpdelta = logpdelta0
    IR = storeIR[int(index[0])]
    LDR = storeLDR[int(index[0])]
    
    for k in range(int(index[0])-1,-1,-1):
        oldcheck = check2
        delta[I] = values[k]
        check = Lambda[1,:] <= delta(Lambda[0,:])
        check2 = np.where(check)
        newcheck = np.setdiff1d(oldcheck,check2)
        L = len(newcheck)
        T = np.identity(L) - np.matrix(W)[:,newcheck].T * IR * np.matrix(W)[:,newcheck]
        T2 = np.matrix(W)[:,newcheck].T * IR
        [IT, LDT] = invandlogdet(T)
        IR = IR + T2.T * IT * T2
        LDR = LDT + LDR
        storeIR[k] = IR
        storeLDR[k] = LDR
        RSS = (Y2.T) * IR * Y2 # Y2 must be a matrix; otherwise use np.matrix(Y2) 
        logpdelta = logpdelta - E(I) * lrho
        m2lprob = n * np.log(RSS) + LDR - 2 * logpdelta
        calculatem2lprob[k] = m2lprob  # Check if it is correct 
  
    return (calculatem2lprob, values, storeIR, storeLDR)
    
    
def calculatem2lprob2(delta): # Needs to be checked for errors
    """
    describtion
    """

    #global W Y2 d n Nf R00 lrho
    #global E Lambda
    
    ¤check = Lambda[1,:] <= delta(Lambda[0,:])
    check2 = np.where(check)
    R = R00 + np.matrix(W)[:,check2] * np.matrix(W)[:,check2].T
    [IR, LDR] = invandlogdet(R)
    storeIR = IR
    storeLDR = LDR
    RSS = (Y2.T) * IR * Y2 # Y2 must be a matrix; otherwise use np.matrix(Y2)
    logpdelta = sum(delta[0][:d] * lrho) + sum((np.maximum(delta[0][d:Nf] - E[d:Nf] + 1, 0)) * E[d:Nf] * lrho)
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
    
def SimulateSobolIndices(model):
    
    global X W Y2 mY d n M2 Nf R00 gamma0 gamma r
    global AnovaIndicators Lambda
    global SobolIndices TotalSensitivityIndices I Type quantile
    global Filename NI
    
    .
    .
    .
    
    
    
    
def CdfDev(x):
    """
    describtion
    """
    global quantile
    CdfDev = CDFEst(x) - quantile
    return CdfDev
    
    
def CDFEst(x):
    """
    describtion
    """
    global SobolIndices TotalSensitivityIndices I Type
    
    if x < 0:
        CDFEst = 0
    elif x >= 1:
        CDFEst = 1
        
    if Type == 1:
        CDFEst = PiecewiseLinearCDF(x,SobolIndices[I]) # needs test for real problem
    elif Tupe == 2:
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
    
    