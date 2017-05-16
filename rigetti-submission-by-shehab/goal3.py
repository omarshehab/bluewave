# Author: Omar Shehab
# Email: shehab1@umbc.edu
# Date: May 4, 2017

# This program uses the Forest API from the Rigetti Quantum Computing
# to solve a few challenge problems as a part of the interviewing process.
# The author has tried to use as few quantum gates as he can.

# This program is written for Goal 3.

# Goal 3: Same as Goal 2, except now the qubits you use are arranged on
# the vertices of an N x N square lattice and the only two-qubit gates
# available happen between neighboring qubits in the lattice, e.g. for
# qubits numbered:

#    6 7 8
#    3 4 5
#    0 1 2

# there is a two qubit gate allowed between (1 and 2) and (5 and 2) but
# not between 2 and any other qubit.

# No need to worry about implementing any kind of measurement on these states.
# Just attempt to create the appropriate objects. Let me know if you have any
# other questions as you approach the solution.


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
import SquareLattice2D


# This function creates a 2D square lattice using the SquareLattice2D class.
def run_2D_square_lattice():
   while True:
      try:
         print "Input the number of qubits in either dimension of the 2D lattice ('q' to quit): "
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

            print "Creating an " + str(n) + " X " + str(n) + " qubit lattice..."

            two_D_square_lattice = SquareLattice2D.SquareLattice2D(n)

            two_D_square_lattice.print_lattice()

            two_D_square_lattice.print_lattice_ghz_state()
      except ValueError:
         print "Oops!  The input has to be an integer.  Please try again..."
      except Exception as ex:
         print "Oops!  Unknown error occured.  Please try again..."
         template = "An exception of type {0} occurred. Arguments:\n{1!r}"
         message = template.format(type(ex).__name__, ex.args)
         print message
   return
   
   
