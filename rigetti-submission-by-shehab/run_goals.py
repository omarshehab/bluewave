# Author: Omar Shehab
# Email: shehab1@umbc.edu
# Date: May 4, 2017

# This program uses the Forest API from the Rigetti Quantum Computing
# to solve a few challenge problems as a part of the interviewing process.
# The author has tried to use as few quantum gates as he can.

# This program automatically run all goals one after another.

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
import goal1
import goal2
import goal3
import goal4



# This function runs all the goals one after another.
def run():
   print "Achieving goal 1:"
   goal1.print_bell_states()

   print "\nAchieving goal 2:"
   goal2.print_ghz_state()

   print "\nAchieving goal 3:"
   goal3.run_2D_square_lattice()

   print "\nAchieving goal 4:"
   goal4.run_qubit_graph()

   return

