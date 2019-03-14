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
connected components of the graph are found by a breadth(deep) ﬁrst search. 
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
    # i.e. if len(cc) >= 2:

    # Draw the re-ordered matrix Mδ as reMdelta
    # cc[i] shows the i-th block
    reOrderRow = []
    reOrderCol = []
    for i in range(len(cc)):
        for j in cc[i]:
            if j < k:
                reOrderRow.append(j)   
            else:
                reOrderCol.append(j-d)
    reMdelta = Mdelta[:,reOrderCol] # Reorder the column of Mδ
    reMdelta = reMdelta[reOrderRow,:] # Reorder the rows of Mδ
    
    return (cc, reMdelta)


