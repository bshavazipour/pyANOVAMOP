# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:44:11 2019

@author: babshava
"""

"""
new implimentation with some changes by Babooshka Shavazipour

Original implimented by Daniele Bigoni (dabi@imm.dtu.dk)
"""

from scipy import stats
import sobol_seq

def scramble(X):
    """
    Scramble function as in Owen (1997)
    
    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """

    N = len(X) - (len(X) % 2)
    
    idx = X[0:N].argsort()
    iidx = idx.argsort()
    
    # Generate binomial values and switch position for the second half of the array
    bi = stats.binom(1,0.5).rvs(size=int(N/2)).astype(bool)
    pos = stats.uniform.rvs(size=int(N/2)).argsort()
    
    # Scramble the indexes
    tmp = idx[0:int(N/2)][bi]
    idx[0:int(N/2)][bi] = idx[int(N/2):N][pos[bi]]
    idx[int(N/2):N][pos[bi]] = tmp
    
    # Apply the scrambling
    X[0:N] = X[0:N][idx[iidx]]
    
    # Apply scrambling to sub intervals
    if N > 2:
        X[0:int(N/2)] = scramble(X[0:int(N/2)])
        X[int(N/2):N] = scramble(X[int(N/2):N])
    
    return X



def scrambled_sobol(d, n):
    """
    Scramble function as in Owen (1997)
    
    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """
    
    # Generate sobol sequence
    samples = sobol_seq.i4_sobol_generate(d, 10**4) # generate a matrix includes 10^4 terms of the Sobol sequence (if it took time reduce it to 10^3)
    
    # Scramble the sequence
    for col in range(0,d):
        samples[:,col] = scramble(samples[:,col]) #Scrambled them in each column
    
    return samples[0:n, :] # return the first n rows of the scrambled matrix
