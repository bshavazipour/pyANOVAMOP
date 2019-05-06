# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:34:57 2019

@author: babshava
"""

"""
Check decomposability(Remark 6):
    
   To ﬁnd the blocks of the decomposition we consider the matrix Mδ as the incidence matrix of a(n) (un)directed
bipartite graph with two lists of d and k nodes, where there exists a connection from
the node i of the ﬁrst list to the node ℓ of the second list if m_i^ℓ = 1. The matrix
has two or more blocks if the graph has two or more connected components. The
connected components of the graph are found by a breadth/deep ﬁrst search. 

It also retun the reordered matrix Mδ if it is decomposible at the second part.
"""

import Graph 

def CheckDecomposability(d, k, Mdelta):
    # building the edges in (undirected) bipartite graph
    g = Graph(d + k);
    for i in range(d):
        for l in range(k):
            if Mdelta[i][l] == 1:
                g.addEdge(i, d + l)            
    cc = g.connectedComponents()          
    print("Following are connected components")
    print(cc)  # cc is a list of lists includes the  connected components of the graph
    # The problem is δ-decomposable if the graph has two or more connected components 
    #(i.e. the corresponding matrix, at least, has two blocks).
    # i.e. if there exsit an i s.t. len(cc[i]) >= 2: (at least to blocks)
    Decomposable = False # If True the Matrix is decomposable
    
    if len(cc) >= 2:
        for i in range(len(cc)):
            if  len(cc[i]) >= 2:
                Decomposable = True

# Alternative if (k+d-1) > len(cc) >= 2
                
    # Draw the re-ordered matrix Mδ as reMdelta
    # cc[i] shows the i-th block
    # Rows are numbered from 0 to k-1, then, if (in corresponding bipartite graph) a vertex from the first group(functions)
    # is connected to a vertex from the second group (variable) it has a value of 1. Otherwise, its value is 0.
    # colomns are numbered from k to k+d-1, then, if (in corresponding bipartite graph) a vertex from the first group(functions)
    if Decomposable:
        reOrderRow = []
        reOrderCol = []
        for i in range(len(cc)):
            for j in cc[i]:  
                if j < k:   # i.e. j is related to the first group (function)
                    reOrderRow.append(j)   # Add it to a new list of rows
                else: # i.e. j is related to the second group (variables)
                    reOrderCol.append(j-d) # Add it to a new list of columns
        reMdelta = Mdelta[:,reOrderCol] # Reorder the column of Mδ
        reMdelta = reMdelta[reOrderRow,:] # Reorder the rows of Mδ
    else:
        reMdelta = []
    
    return (cc, Decomposable, reMdelta)



# separating the blocks 
for i in range(len(cc)):
    for j in cc[i]:
        fd[i][j] = 


"""
An example from the paper(p3264 FIG 1.) to check the function.

A = np.array([[0,0,1,0,0,1,0,1],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,1,0,0,0],
              [0,0,1,0,0,1,0,0],
              [1,0,0,1,0,0,1,0],
              [0,1,0,0,1,0,0,0],
              [0,0,0,0,0,1,0,1],
              [1,0,0,1,0,0,0,0]])
    
"""
