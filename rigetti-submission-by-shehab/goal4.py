# Author: Omar Shehab
# Email: shehab1@umbc.edu
# Date: May 4, 2017

# This program uses the Forest API from the Rigetti Quantum Computing
# to solve a few challenge problems as a part of the interviewing process.
# The author has tried to use as few quantum gates as he can.

# This program is written for Goal 4.

# Goal 4: Generalization.

# Consider a collection of qubits in a quantum computer. Due to locality
# constraints, two-qubit gates can often only be applied to neighboring
# qubits, for some notion of "neighboring". This leads to a graph
# structure, where edges of this graph represent where two-qubit gates
# can be applied.

# Using pyQuil, write a function to produce the |V|-qubit
# state

#    (|000...0> + |111...1>) / sqrt(2)

# on a graph G = (V, E).



from pyquil.gates import CNOT
from pyquil.gates import H
from pyquil.gates import X
import numpy as np
from pyquil.quil import Program
import pyquil.forest as forest
quantum_simulator = forest.Connection()
from pyquil.gates import I
import itertools
import goal_util
import QGraph
import random
import networkx as nx



# This function creates a planar lattice using the input graph connectivity.
def run_qubit_graph():
   while True:
      try:
         print "Input the number of nodes in the graph ('q' to quit): "
         n_in = raw_input()

         if n_in == 'q':
            print "Terminating the program..."
            return
         else:
            n = int(n_in)
            # Making sure that n is a nonzero positive integer.
            if n < 1:
               print "The number of qubits has to be at least 1."
               continue

            # edge_probability = random.random() 
            # print "Creating a " + str(n) + "-node random graph with edge probability: " + str(edge_probability)
            # qgraph = QGraph.QGraph(nx.erdos_renyi_graph(n, edge_probability))
            # qgraph.get_linear_chain_from_graph()

            print "Creating a " + str(n) + "-node comlete graph..."
            qgraph = QGraph.QGraph(nx.complete_graph(n))
            qgraph.print_ghz_state_from_complete_graph()

            print "Creating a " + str(n) + "-node star graph..."
            qgraph = QGraph.QGraph(nx.star_graph(n - 1))
            qgraph.print_ghz_state_from_star_graph()

      except ValueError:
         print "Oops!  The input has to be an integer.  Please try again..."
      except Exception as ex:
         print "Oops!  Unknown error occured.  Please try again..."
         template = "An exception of type {0} occurred. Arguments:\n{1!r}"
         message = template.format(type(ex).__name__, ex.args)
         print message
   return

