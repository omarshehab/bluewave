import networkx

def generate():
   nodes = 100
   probability = 0.5
   print "Creating graphs with " + str(nodes) + " and probability " + str(probability)
   graph = erdos_renyi_graph(nodes, probability)

