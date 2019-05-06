# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:00:32 2019

@author: babshava

Ref: https://www.geeksforgeeks.org/cartesian-product-of-any-number-of-sets/
"""

# 

"""
Cartesian Product of two sets
"""
def Cartesian_Product(set_a, set_b): 
    result =[] 
    for i in range(0, len(set_a)): 
        for j in range(0, len(set_b)): 
  
            # for handling case having cartesian 
            # prodct first time of two sets 
            if type(set_a[i]) != list:          
                set_a[i] = [set_a[i]] 
                  
            # coping all the members 
            # of set_a to temp 
            temp = [num for num in set_a[i]] 
              
            # add member of set_b to  
            # temp to have cartesian product      
            temp.append(set_b[j])              
            result.append(temp)   
              
    return result 

# Function to do a cartesian  
# product of N sets  
def Cartesian_Product_of_m(list_a):  # list_a must be a set or a list of tuples
      
    # result of cartesian product 
    # of all the sets taken two at a time 
    temp = list_a[0] 
      
    # do product of N sets  
    for i in range(1, len(list_a)): 
        temp = Cartesian_Product(temp, list_a[i]) 
    
    print(temp)             
    return temp
    
"""
e.g. 
A = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
B = [(4, 4), (5, 5)]
C = [(6, 6), (7, 7)]
p = [A, B, C]

P = Cartesian_Product_of_m(p)

Out: 
[[(1, 1, 1), (4, 4), (6, 6)],
 [(1, 1, 1), (4, 4), (7, 7)],
 [(1, 1, 1), (5, 5), (6, 6)],
 [(1, 1, 1), (5, 5), (7, 7)],
 [(2, 2, 2), (4, 4), (6, 6)],
 [(2, 2, 2), (4, 4), (7, 7)],
 [(2, 2, 2), (5, 5), (6, 6)],
 [(2, 2, 2), (5, 5), (7, 7)],
 [(3, 3, 3), (4, 4), (6, 6)],
 [(3, 3, 3), (4, 4), (7, 7)],
 [(3, 3, 3), (5, 5), (6, 6)],
 [(3, 3, 3), (5, 5), (7, 7)]]

"""  
    

"""
# Driver Code 
list_a = [[1, 2],          # set-1 
          ['A'],          # set-2 
          ['x', 'y', 'z']]   # set-3 
            
  
# Function is called to perform  
# the cartesian product on list_a of size n  
Cartesian_Product_of_m(list_a)
"""






# ---------------------------------------------------------------------------------------


"""
Cartesian Product of two tuples
"""

def cartesian_product(tup1, tup2):
    """Returns a tuple that is the Cartesian product of tup_1 and tup_2

    >>> X = (1, 2, 3)
    >>> Y = (4, 5)
    >>> cartesian_product(X, Y)
    ((1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5))
   
    """
    return tuple((t1, t2) for t1 in tup1 for t2 in tup2)

"""
Cartesian Product of m tuples

input is a tuples with m member (p) [includes the solutions of the subproblems p_1, p_2, ..., p_3]

Output is the cartesian product
"""
#p = []
#for i in range(m):
#    p.append()

def cartesian_productm(p): # p is a list & m is the number of sets
    """Returns the cartesian product of all the m sets taken two at a time 

    >>> p1 = (x1, x2, x3)
    >>> p2 = (x4, x5)
    >>> ...
    >>> p_m = (xm)
    >>> cartesian_product(p1, p2,...,pm)
    ((x1, x4, ..., xm), (x3, x5, ..., xm))
   
    """
    p1 = p[0]
    P = []
    for i in range(1, len(p)):
        p1 = cartesian_product(p1, p[i])
        
    
    return p1 # tuple


