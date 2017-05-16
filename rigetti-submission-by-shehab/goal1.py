# Author: Omar Shehab
# Email: shehab1@umbc.edu
# Date: May 4, 2017

# This program uses the Forest API from the Rigetti Quantum Computing
# to solve a few challenge problems as a part of the interviewing process.
# The author has tried to use as few quantum gates as he can.

# This program is written for Goal 1.

# Goal 1: Output one of the four Bell states. 


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


# This function gives the users a menu of four choices for four Bell states.
def print_bell_states():
   while True:
      try:
         print "Please choose the Bell state you want to print: \n1. Phi^+\n2. Phi^-\n3. Psi^+\n4. Psi^-\n5. Quit"
         bell_state_index = input()

         if bell_state_index == 1:
            print "To generate Phi^+ we have to apply the gate on |00>."
            # Flipping nothing
            p = Program(I(0), I(1), H(0), CNOT(0, 1))
            goal_util.print_wavefunction_details(p)
         elif bell_state_index == 2:
            print "To generate Phi^- we have to apply the gate on |10>."
            # Flipping the second qubit
            p = Program(X(0), I(1), H(0), CNOT(0, 1))
            goal_util.print_wavefunction_details(p)
         elif bell_state_index == 3:
            print "To generate Psi^+ we have to apply the gate on |01>."
            # Flipping the first qubit
            p = Program(I(0), X(1), H(0), CNOT(0, 1))
            goal_util.print_wavefunction_details(p)
         elif bell_state_index == 4:
            print "To generate Psi^- we have to apply the gate on |11>."
            # Flipping both qubits
            p = Program(X(0), X(1), H(0), CNOT(0, 1))
            goal_util.print_wavefunction_details(p)
         elif bell_state_index == 5:
            print "Terminating the program..."
            return
         else:
            print "Please provide an input between 1 to 5."
      except SyntaxError:
         print "Oops!  That was no valid input.  Please try again..."
      except Exception as ex:
         print "Oops!  Unknown error occured.  Please try again..."
         # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
         # message = template.format(type(ex).__name__, ex.args)
         # print message
   return


