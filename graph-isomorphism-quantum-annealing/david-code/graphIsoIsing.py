import networkx as nx
from numpy import *


def matrix_to_dict(matrix):

    # Determine the dimension of the matrix
    length = len(matrix)

    # Declare the dictionary
    memory_dict = {}

    # Initialize it with zero
    for i in range(0, length):
        for j in range(0, length):
            memory_dict[(i, j)] = 0.0

    # Assign the values
    for i in range(0, length):
        for j in range(0, length):
            if j < i:
                memory_dict[(i, j)] = matrix[i][j]
    return memory_dict

def tupleOrder(x):
    if x[0] > x[1]:
        return((x[1], x[0]))
    else:
        return(x)
    

# Inputs: two graph objects from networkx, a boolean
# true if return as ising, false if return as qubo, a bool,
# true if print matrix, false otherwise

# Outputs: a list, with first element a list of the diagonal
# elements (h), and the second element a dictionary with
# off diagonal elements, where the key is the mapping of vertices (i,u)
# and the value is the associated spin variable, J_(i,u)
def createIsing(graph1, graph2, verbose):
    
    # If there are differing numbers of isolated nodes, throw
    # an exception, cannot be isomorphic
    if(len(nx.isolates(graph1)) != len(nx.isolates(graph2))):
        raise Exception("Graphs have different number of vertices of degree zero")
    
    
    # Create a list with all parings of nodes that have
    # the same degree
    setOfTwo = []
    for i in range(len(graph1.nodes())):
        for j in range(len(graph2.nodes())):
            
            # exclude degree zero vertices
            if graph1.degree()[i] == graph2.degree()[j] and graph2.degree()[j] > 0:
                setOfTwo.append((i,j))
    print "setOfTwo: "
    print str(setOfTwo)                       

    #Initialize result matrix
    resultMat = zeros(shape = (len(setOfTwo), len(setOfTwo)))

    # go through possible valid pairings
    for i in range(len(setOfTwo)):
        gr1 = setOfTwo[i]
        print "gr1: " + str(gr1)

        for j in range(len(setOfTwo)):
            gr2 = setOfTwo[j]
            print "gr2: " + str(gr2)
                
            # puts edges in order so that
            # smaller value is first
            edge1 = tupleOrder((gr1[0],gr2[0]))
            print "edge1: " + str(edge1)
            edge2 = tupleOrder((gr1[1],gr2[1]))
            print "edge2: " + str(edge2)
            
            # If diagonal element
            if i == j:
                print "i: " + str(i) + " == " + "j: " + str(j)
                resultMat[i][j] = 0
            
            # if i == j, penalize mapping
            elif gr1[0] == gr2[0]:
                print "gr1[0] == gr2[0]: " + str(gr1[0]) + " == " + str(gr2[0])
                resultMat[i][j] = 1
                
            # if u == v, penalize
            elif gr1[1] == gr2[1]:
                print "gr1[1] == gr2[1]: " + str(gr1[1]) + " == " + str(gr2[1])
                resultMat[i][j] = 1
                
                
            # if (i,j) is an edge, but not (u,v), penalize mapping
            elif edge1 in graph1.edges() and edge2 not in graph2.edges():
                print "edge1 " + str(edge1) + " in " + str(graph1.edges())
                print "edge2 " + str(edge2) +  " not in " + str(graph2.edges())
                resultMat[i][j] = 1
                
            # if (u,v) is an edge,  but not (i,j), penalize mapping
            elif edge1 not in graph1.edges() and edge2 in graph2.edges():
                print "edge1 " + str(edge1) + " not in " + str(graph1.edges())
                print "edge2 " + str(edge2) +  " in " + str(graph2.edges())
                resultMat[i][j] = 1
                
            else:
                resultMat[i][j] = 0
                
        # print QBO
        print "Printing current QUBO: "
        print(triu(resultMat))
        
    # get diagonal elements as -1 b4 change to ising
    diag = resultMat.diagonal()
    
    # print QBO
    print "Printing QUBO: "
    print(triu(resultMat))
    
    # convert QUBO to ising
    resultMat = (2 * resultMat) - 1
    
    
    if verbose:
	print "Printing Ising: "
        print(triu(resultMat))
        
    
    return([diag, matrix_to_dict(resultMat)])
      
