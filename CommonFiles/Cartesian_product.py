# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:00:32 2019

@author: babshava
"""

"""
Cartesian Product of two tuples
"""

def cartesian_product(tup1, tup2):
    """Returns a tuple that is the Cartesian product of tup_1 and tup_2

    >>> X = (1, 2)
    >>> Y = (4, 5)
    >>> cartesian_product(X, Y)
    ((1, 4), (1, 5), (2, 4), (2, 5))
    """
    return tuple((t1, t2) for t1 in tup1 for t2 in tup2)