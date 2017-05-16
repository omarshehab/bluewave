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
import sys


class SquareLattice2D:
   # Constructor takes the N as input
   def __init__(self, dimension):

      # Making sure that is a nonzero positive integer.
      if dimension < 1:
         print "The number of qubits has to be at least 1."
         sys.exit()

      print "Creating " + str(dimension) + "-qubit square lattice states..."

      self.dimension = dimension
      self.program = Program()

   # This function applies a CNOT gate on the (row1, column1) and (row2, column2)-th qubits.
   def apply_CNOT_on_lattice_site(self, row1, column1, row2, column2):

      # Applying gate on the (row1, column1) and (row2, column2)-th qubits
      linear_index1 = self.get_linear_qubit_index_from_2D_coordinate(row1, column1)
      linear_index2 = self.get_linear_qubit_index_from_2D_coordinate(row2, column2)

      # Checking if these two qubits are connected.
      if (not self.are_connected((row1, column1), (row2, column2))):
         # print "Invalid pairs"
         return False

      self.program.inst(CNOT(linear_index1, linear_index2))

      return True

   # This function applies an H gate on the (row, column)-th qubit.
   def apply_H_on_lattice_site(self, row, column):

      # Applying gate on the (row, column)-th qubit
      linear_index = self.get_linear_qubit_index_from_2D_coordinate(row, column)

      self.program.inst(H(linear_index))
      return True

   # Create a linear chain of nearest neighbors from a square lattice.
   def get_linear_chain_from_square_lattice(self):

      chain_of_pairs = []
      chain_of_qubits = []

      start_qubit_of_the_row = 0
      final_qubit_of_the_row = self.dimension
      step_of_increment = 1

      finished_row_traversing = False
      final_row_traversed = False

      for row in range(self.dimension):
         for column in range(start_qubit_of_the_row, final_qubit_of_the_row, step_of_increment):
            chain_of_pairs.append((row, column))
            chain_of_qubits.append(self.get_linear_qubit_index_from_2D_coordinate(row, column))

            if ((column % self.dimension) == (self.dimension - 1)):
               if not finished_row_traversing:
                  finished_row_traversing = True

                  if row == self.dimension - 1:
                     final_row_traversed = True

                     return chain_of_pairs, chain_of_qubits

                  row = row + 1
                  if row == self.dimension - 1:
                     final_row_traversing = True

                  start_qubit_of_the_row = self.dimension - 1
                  final_qubit_of_the_row = -1
                  step_of_increment = -1

                  continue
               else:
                  finished_row_traversing = False
                  continue
            elif (row > 0) and ((column % self.dimension) == 0):
               if not finished_row_traversing:
                  finished_row_traversing = True

                  if row == self.dimension - 1:
                     final_row_traversed = True

                     return chain_of_pairs, chain_of_qubits

                  row = row + 1

                  start_qubit_of_the_row = 0

                  final_qubit_of_the_row = self.dimension

                  step_of_increment = 1

                  continue
               else:
                  finished_row_traversing = False

                  continue
            else:
               print ""


            

   # Checks if the input qubit pair is a connected pair.
   def are_connected(self, qubit1, qubit2):
      # Basic validation to make sure that the coordinates are between 0 and n
      if qubit1[0] < 0 or qubit1[0] >= self.dimension or qubit1[1] < 0 or qubit1[1] >= self.dimension or qubit2[0] < 0 or qubit2[0] >= self.dimension or qubit2[1] < 0 or qubit2[1] >= self.dimension:
         print "Coordinates out of range."
         return False

      if (abs(qubit1[0] - qubit2[0]) == 1 and qubit1[1] == qubit2[1]) or (abs(qubit1[1] - qubit2[1]) == 1 and qubit1[0] == qubit2[0]):
         return True
      else:
         return False

   # Printing the latice state with probability.
   def print_lattice_state(self, basis_state, probability):
      print "Print lattice state..."
      lattice = []

      # Location of one in the basis state.
      location_of_one = [i for i, letter in enumerate(basis_state) if letter == '1']

      for row_count in range(self.dimension):
         row = ["|1>" if basis_state[self.get_linear_qubit_index_from_2D_coordinate(row_count, x)] == '1' else "|0>" for x in list(range(self.dimension))]
         lattice.append(row)

      for row in reversed(lattice):
         print('\t'.join(map(str, row)))

      print "Probability : " + str(probability)

   # Printing the latice.
   def print_lattice(self):
      lattice = [[]]

      for row_count in range(0, (self.dimension**2), self.dimension):
         # print "Row: " + str(row_count)
         row = [x + row_count for x in list(range(self.dimension))]
         lattice.append(row)

      for row in reversed(lattice):
         print('\t'.join(map(str, row)))
      

   # Return the 2D coordinates of the i-th qubit. The coordinate of the
   # bottom left qubit is 0,0.
   def get_2D_coordinate_from_linear_index(self, qubit):
      column = qubit % self.dimension
      row = qubit / self.dimension

      return (row, column)


   # Return the linear qubit index from 2D coordinates. The coordinate of the
   # bottom left qubit is 0,0.
   def get_linear_qubit_index_from_2D_coordinate(self, row, column):
      index = (row * self.dimension) + column

      return index

   # Checking if the current qubit is on the boundary.
   def is_boundary_qubit(self, qubit):

      # List of qubits on the right boundary

      right_boundary = [y + (self.dimension - 1) for y in [x * self.dimension for x in list(range(self.dimension))]]

      # List of qubits on the left boundary
      left_boundary = [x * self.dimension for x in list(range(self.dimension))]
      

      # Checking if the qubit is at the bottom edge.
      if qubit >= 0 and qubit < self.dimension:
         return True

      # Checking if the qubitis at the top edge.
      elif qubit >= (self.dimension * (self.dimension - 1 )) and qubit < ((self.dimension * self.dimension) - 1):
         return True

      # Checking if the qubitis at the left or right edges.
      elif qubit in left_boundary or qubit in right_boundary:
         return True

      else:
         return False

   def print_lattice_ghz_state(self):
      # Get all qubits in linear chain order.
      chain = self.get_linear_chain_from_square_lattice()
      
      chain_length = len(chain[1])

      # Apply Hadamard gate on the first qubit
      p = Program(H(0))

      # Run a loop to apply CNOT gate on all the consecutive pairs.
      for qubit1 in range(chain_length - 1):
         p.inst(CNOT(qubit1, qubit1 + 1))

      # Print the details of the state
      basis_states, amplitudes, probabilities = goal_util.print_wavefunction_details(p)

      amplitudes.reset()
      probabilities.reset()

      for probability in probabilities:
         if probability == 0.0:
            continue

         self.print_lattice_state(basis_states[probabilities.iterindex], probability)            
         print "\n"

      return

