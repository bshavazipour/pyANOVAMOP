# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:07:36 2019

@author: babshava
"""

"""
 Build the reduced incidence matrix Mδ with  ------------------------------
 ---------------------------------------------------------------------------------
 Note: If δ = 0 the incidence matrix M will be build, otherwise (0<δ<1) the Mδ. 
 However, in both case we name it Mδ here.
 
 M := [m_i^l]_{l,i} s.t \{ m_i^l = 1; if x_i is active (T_i^l > 0)
                        \{ m_i^l = 0; if x_i is inactive (T_i^l = 0)
 ----------------------------------------------------------------------------------
 Mδ := [m_i^l]_{l,i} s.t \{ m_i^l = 1; if T_i^l >= δ
                         \{ m_i^l = 0; if T_i^l < δ
 where k is the number of objective functons, d is the number of variables and δ is threshold.
"""
import numpy as np
 
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