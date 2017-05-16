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
from pyquil.gates import Z
from pyquil.gates import CPHASE
import numpy as np
from pyquil.quil import Program
import pyquil.forest as forest
quantum_simulator = forest.Connection()
from pyquil.gates import I
import itertools
import goal_util
import sys
import networkx as nx
from networkx.utils import uniform_sequence, create_degree_sequence
import collections
import matplotlib.pyplot as plt
import goal2
import operator


# QGraph object represents a graph state.
# In future it will also be extended to
# cluster states. It provides API to apply
# H and CNOT gates based on graph connectivity.
# In future, a general method will be included
# to apply arbitray gates based on graph connectivity.
class QGraph:
   # Constructor takes a networkx graph as input
   def __init__(self, graph):

      # Currently we support only the NetworkX graph objects.
      if not isinstance(graph, nx.classes.graph.Graph):
         print "The input is not a NetworkX graph."
         raise Exception

      self.graph = graph

      # For a GHZ graph state, self.is_star is True when it is a star graph and 
      # False when it is a complete graph.
      self.is_star = False
      self.is_complete = False

      self.n = nx.number_of_nodes(self.graph)

      # Checking if it is a complete graph.
      # print "Checking if it is a complete graph."
      if not len(self.graph.edges()) == ((self.n * (self.n - 1)) / 2):
         print "The input is not a complete graph."
      else:
         self.is_complete = True
         self.is_star = False
         print "It is a complete graph."

      # Checking if it is a star graph.
      # print "Checking if it is a star graph."
      if ((self.get_number_of_degree_one_nodes(self.graph) + self.get_number_of_degree_n_1_nodes(self.graph)) != self.n) or self.get_number_of_degree_n_1_nodes(self.graph) != 1:
         print "The input is not a star graph."
      else:
         self.is_star = True
         self.is_complete = False
         print "It is a star graph."
   
      # If the graph is neither a star nor a complete,
      # a GHZ state cannot be created.
      if not self.is_star and not self.is_complete:
         print "The graph is neither a star and nor a complete."
         raise Exception("The graph is neither a star and nor a complete.")

      print str(nx.adjacency_matrix(self.graph))

      self.program = Program()


   # This function returns the central node of a star graph.
   def get_central_node_from_star_graph(self):

      center = max(nx.degree(self.graph).iteritems(), key=operator.itemgetter(1))[0]
      # print "Center node is: " + str(center)
      return center


   # This function creates a GHZ state on a star graph.
   # It implements the algorithm given in http://www.ma.rhul.ac.uk/akay/technicalities/multipartite.php
   # by Alastair Kay.
   #
   # Not working hence not used.
   def print_ghz_state_from_star_graph_kay(self):

      for qubit in range(self.n):
         neighbors = self.graph.neighbors(qubit)

         for neighbor in range(len(neighbors)):
            self.program.inst(X(qubit), Z(neighbor))

      goal_util.print_wavefunction_details(self.program)

      return True


   # This function creates a GHZ state on a star graph.
   # It implements the algorithm given in Figure 8
   # of Performing Quantum Computing Experiments in the Cloud
   # by Simon J. Devitt (arXiv: 1605.05709).
   #
   # Not working hence not used.
   def print_ghz_state_from_star_graph_devitt(self):
      center = self.get_central_node_from_star_graph()
      
      for qubit in range(self.n):

         if qubit == center:
            continue
         else:
            self.program.inst(H(qubit))

      goal_util.print_wavefunction_details(self.program)

      for qubit in range(self.n):

         if qubit == center:
            continue
         else:
            self.program.inst(CNOT(qubit, center))

      goal_util.print_wavefunction_details(self.program)

      return True

   # This function creates a GHZ state on a star graph.
   # 
   # Omar Shehab's naive approach which happens to work.
   # Working.
  
   # This function puts the center node into an equal 
   # superposition and CNOTS the connected nodes based on it.
   def print_ghz_state_from_star_graph(self):

      # Identify the center node.
      center = self.get_central_node_from_star_graph()
      print center

      # Apply H on it.
      self.program.inst(H(center))

      # Get all the neighbors.
      neighbors = self.graph.neighbors(center)

      # Apply CNOT on all neighbors with center as the control bit.
      for neighbor in neighbors:
         print center
         print neighbor
         self.program.inst(CNOT(center, neighbor))

      # Print the details of the state
      goal_util.print_wavefunction_details(self.program)

      return True



   # This function creates a GHZ state on a complete graph.
   # If it is a complete graph, we just identify a linear chain 
   # and create a GHZ state using goal2.
   def print_ghz_state_from_complete_graph(self):

      # All possible interactions on the graph are available
      # so we recycling the function to create it on a linear chain.
      self.program = goal2.get_ghz_program_with_arbitrary_connections(self.n)

      # Print the details of the state
      goal_util.print_wavefunction_details(self.program)
      return True


   # This function applies a CNOT gate on the node1 and node2-th nodes.
   def apply_CNOT_on_nodes(self, node1, node2):
      # Checking if the node is connected.
      if sorted((node1, node2)) not in self.graph.edges():
         return False

      self.program.inst(CNOT(node1, node2))

      return True

   # This function applies an H gate on the node-th qubit.
   def apply_H_on_lattice_site(self, node):
      # Applying gate on the node-th qubit
      self.program.inst(H(node))
      return True

   # Get degree sequence from graph.
   def get_number_of_degree_one_nodes(self, graph):
      number_of_degree_one_nodes = sum(1 for x in nx.degree(graph).values() if x == 1) 
      # print "Number of degree one nodes: " + str(number_of_degree_one_nodes)
      return number_of_degree_one_nodes

   # Get degree sequence from graph.
   def get_number_of_degree_n_1_nodes(self, graph):
      number_of_degree_n_1_nodes = sum(1 for x in nx.degree(graph).values() if x == (nx.number_of_nodes(graph) - 1)) 
      # print "Number of degree n-1 nodes: " + str(number_of_degree_n_1_nodes)
      return number_of_degree_n_1_nodes
      

