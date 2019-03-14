# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:35:17 2019

@author: babshava
"""

"""
Source: https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/

Connected Components in an undirected graph
Given an undirected graph, print all connected components line by line. 

Finding connected components for an undirected graph is an easier task. 
We simple need to do either BFS or DFS starting from every unvisited vertex, 
and we get all strongly connected components. Below are steps based on DFS.

1) Initialize all vertices as not visited.
2) Do following for every vertex 'v'.
       (a) If 'v' is not visited before, call DFSUtil(v)
       (b) Print new line character

DFSUtil(v)
1) Mark 'v' as visited.
2) Print 'v'
3) Do following for every adjacent 'u' of 'v'.
     If 'u' is not visited, then recursively call DFSUtil(u)
"""
# Python program to print connected  
# components in an undirected graph 
class Graph: 
      
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)] 
  
    def DFSUtil(self, temp, v, visited): 
  
        # Mark the current vertex as visited 
        visited[v] = True
  
        # Store the vertex to list 
        temp.append(v) 
  
        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 
  
    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 
        
        
 # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc 
 
    """
    example:
        
# Driver Code 
if __name__=="__main__": 
      
    # Create a graph given in the above diagram 
    # 5 vertices numbered from 0 to 4 
    g = Graph(5); 
    g.addEdge(1, 0) 
    g.addEdge(2, 3) 
    g.addEdge(3, 4) 
    cc = g.connectedComponents() 
    print("Following are connected components") 
    print(cc) 
  
# This code is contributed by Abhishek Valsan     
"""