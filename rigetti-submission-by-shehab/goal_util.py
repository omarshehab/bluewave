# Author: Omar Shehab
# Email: shehab1@umbc.edu
# Date: May 4, 2017

# This program uses the Forest API from the Rigetti Quantum Computing
# to solve a few challenge problems as a part of the interviewing process.
# The author has tried to use as few quantum gates as he can.

# This program is written as a utility module for all goals.


from pyquil.gates import CNOT
from pyquil.gates import H
from pyquil.gates import X
import numpy as np
from pyquil.quil import Program
import pyquil.forest as forest
quantum_simulator = forest.Connection()
from pyquil.gates import I
import itertools

# This function prints the probability amplitudes and the probabilities
# of measurement in the computational bases.
def print_wavefunction_details(program):
   # Get the number of qubits
   num_qubits = len(program.extract_qubits())

   # Get the number of computational bases.
   num_computational_bases = len(quantum_simulator.wavefunction(program)[0])   

   # Create the labels of basis states
   basis_states = ["".join(seq) for seq in itertools.product("01", repeat = num_qubits)]

   # Get the wavefunction
   wavefunc, _ = quantum_simulator.wavefunction(program)

   # Print probability amplitudes
   print "The qubits are in the state (probability amplitudes of computational bases in the increasing order of the basis labels): "
   amplitudes = np.nditer(wavefunc)
   for amplitude in amplitudes:
      print "The probability amplitude of the state |" + str(basis_states[amplitudes.iterindex]) + "> is " + str(amplitude)

   print "\n"

   #Print probabilities of measuring qubits in computational bases
   wavefunc_probabilities = abs(wavefunc)**2
   print "The probabilities of measuring the qubits in computational bases are (in the increasing order of the basis labels): "
   probabilities = np.nditer(wavefunc_probabilities)
   for probability in probabilities:
      print "The probability of measuring the state in outcome |" + str(basis_states[probabilities.iterindex]) + "> is " + str(probability)
   
   print "\n"
   return basis_states, amplitudes, probabilities

