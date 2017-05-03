from graphIsoIsing import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as ppl
from matplotlib import pylab


# Random graps
# graph1  = nx.fast_gnp_random_graph(4,.5)
# graph2 = nx.fast_gnp_random_graph(4,.5)

# print "Graph 1: " + str(nx.adjacency_matrix(graph1))
# print "Graph 2: " + str(nx.adjacency_matrix(graph2))


# Graph from the Zick-Shehab paper
testG1 = nx.Graph()
testG1.add_nodes_from(range(0,4))
testG1.add_edges_from([(0,1),(0,2)])

testG2 = nx.Graph()
testG2.add_nodes_from(range(0,4))
testG2.add_edges_from([(0,3),(1,3)])

print "Test Graph 1: " + str(nx.adjacency_matrix(testG1))
print "Test Graph 2: " + str(nx.adjacency_matrix(testG2))

graph1 = testG1
graph2 = testG2

# Settings for graph drawings
node_size_value = 2400
node_color_value = 'b'
font_size_value = 32

# Print graph
print "Saving the NX graph as image..."

ppl.figure(num=None, figsize=(20, 20), dpi=80)
ppl.axis('off')
fig = ppl.figure(1)

print "Creating layout..."

pos = nx.random_layout(graph1)

print "Layout created..."

nx.draw_networkx_nodes(graph1,pos, node_color = node_color_value, node_size = node_size_value)
nx.draw_networkx_edges(graph1,pos)
nx.draw_networkx_labels(graph1, pos, font_size=font_size_value)

cut = 1.00
xmax = cut * max(xx for xx, yy in pos.values())
ymax = cut * max(yy for xx, yy in pos.values())
ppl.xlim(0, xmax)
ppl.ylim(0, ymax)

print "Saving file"

ppl.savefig("g/graph1_rl.png")
# l.savefig("g/graph1_spl.png", bbox_inches="tight")
pylab.close()
del fig
# Print graph

# Print graph
print "Saving the NX graph as image..."

ppl.figure(num=None, figsize=(20, 20), dpi=80)
ppl.axis('off')
fig = ppl.figure(1)

print "Creating layout..."

pos = nx.random_layout(graph2)

print "Layout created..."
nx.draw_networkx_nodes(graph2,pos, node_color = node_color_value, node_size = node_size_value)
nx.draw_networkx_edges(graph2,pos)
nx.draw_networkx_labels(graph2,pos, font_size=font_size_value)

cut = 1.00
xmax = cut * max(xx for xx, yy in pos.values())
ymax = cut * max(yy for xx, yy in pos.values())
ppl.xlim(0, xmax)
ppl.ylim(0, ymax)

print "Saving file"

ppl.savefig("g/graph2_rl.png")
# ppl.savefig("g/graph2_spl.png", bbox_inches="tight")
pylab.close()
del fig
# Print graph


try:
    print(createIsing(graph1, graph2, True))
except:
    print("Non-isomorphic; different number of isolates")

isingProblem = createIsing(testG1, testG2, True)

print("Diagonal Elements: \n", isingProblem[0])
print("Spin Variables: \n", isingProblem[1])


