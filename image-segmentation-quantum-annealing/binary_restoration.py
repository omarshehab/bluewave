from PIL import Image
# import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import networkx as nx
import numpy as np
import scipy
from scipy.misc import imread
from matplotlib import pyplot as ppl
from matplotlib import pylab

import maxflow

# image_file = "halem-noise2"
# image_file = "g/gsp/10"
# image_file = "g/gn/a2"
image_file = "g/org/a2"

print "Reading image as an array: "
img = Image.open(open(image_file + ".png", 'rb'))
print "Image as an array: " + str(img)

print "Shape of the image: " + str(img.shape)

# Create the graph.
g = maxflow.Graph[int](0, 0)
print "The graph: " + str(g)

# Add the nodes.
nodeids = g.add_grid_nodes(img.shape)
print "nodeids: " + str(nodeids)

# Add edges with the same capacities.
g.add_grid_edges(nodeids, 50)
print "Graph after adding edges: " + str(g)

# Add the terminal edges.
g.add_grid_tedges(nodeids, img, 255-img)
print "Graph after adding terminal edges: " + str(g)

graph = g.get_nx_graph()
print "NetworkX graph: " + str(graph)
print "Nodes # " + str(nx.number_of_nodes(graph)) + ", edges # " + str(nx.number_of_edges(graph))
print "Density: " + str(nx.density(graph))
print "Info: " + str(nx.info(graph))
print "Degree histogram: " + str(nx.degree_histogram(graph))
print "Degrees: " + str(sorted(nx.degree(graph).values()))
print "Adjacency matrix: " + str(nx.adjacency_matrix(graph)) 
print "Edge lists: " + str(nx.generate_edgelist(graph))

# Print graph
print "Saving the NX graph as image..."

ppl.figure(num=None, figsize=(20, 20), dpi=1200)
ppl.axis('off')
fig = ppl.figure(1)

print "Creating layout..."

pos = nx.random_layout(graph)

print "Layout created..."
nx.draw_networkx_nodes(graph,pos)
nx.draw_networkx_edges(graph,pos)
nx.draw_networkx_labels(graph,pos)

cut = 1.00
xmax = cut * max(xx for xx, yy in pos.values())
ymax = cut * max(yy for xx, yy in pos.values())
ppl.xlim(0, xmax)
ppl.ylim(0, ymax)

print "Saving file"

ppl.savefig(image_file + "-graph.png", bbox_inches="tight")
pylab.close()
del fig
# Print graph

# Find the maximum flow.
print "Findig the max flow..."
g.maxflow()
# Get the segments.
sgm = g.get_grid_segments(nodeids)
print "Segments: " + str(sgm)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
print "Contents of img2: " + str(img2)
result = Image.fromarray((img2 * 255).astype(np.uint8))
result.save(image_file + "-restored.png")

# Show the result.
# ppl.imshow(img2, cmap=ppl.cm.gray, interpolation='nearest')
# ppl.show()
