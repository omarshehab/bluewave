# Author: Omar Shehab
# Email: shehab1@umbc.edu
# Date: May 4, 2017

# This program uses the Forest API from the Rigetti Quantum Computing
# to solve a few challenge problems as a part of the interviewing process.
# The author has tried to use as few quantum gates as he can.

# This program is written for Goal 2.

# Goal 2: Take as input an integer N. Output an N-qubit state

#    (|000...0> + |111...1>) / sqrt(2)

# This program uses the first protocol mentioned in Neeley, Matthew, et al. "Generation of three-qubit entangled states using superconducting phase qubits." Nature 467.7315 (2010): 570-573.


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


# This function takes an integer N as input from the user and output an
# N-qubit GHZ state.
def print_ghz_state():
   while True:
      try:
         print "Please input the number of qubits for the GHZ state ('q' to quit)."
         ghz_qubits_in = raw_input()

         if ghz_qubits_in == 'q':
            print "Terminating the program..."
            return
         else:
            ghz_qubits = int(ghz_qubits_in)
            # Making sure that ghz_qubits is a nonzero positive integer.
            if ghz_qubits < 1:
               print "The number of qubits has to be at least 1."
               continue
            print "Creating " + str(ghz_qubits) + "-qubit GHZ states..."

            # Print the details of the state
            goal_util.print_wavefunction_details(get_ghz_program_with_arbitrary_connections(ghz_qubits))
      except ValueError:
         print "Oops!  The input has to be an integer.  Please try again..."
      except Exception as ex:
         print "Oops!  Unknown error occured.  Please try again..."
         # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
         # message = template.format(type(ex).__name__, ex.args)
         # print message
   return

# This function assumes all possible connections between the qubits are available
# on the chip.
def get_ghz_program_with_arbitrary_connections(qubits):
   # Apply Hadamard gate on the first qubit
   prg = Program(H(0))

   # Run a loop to apply CNOT gate on all the consecutive pairs.
   for qubit1 in range(qubits - 1):
      prg.inst(CNOT(qubit1, qubit1 + 1))
   return prg

