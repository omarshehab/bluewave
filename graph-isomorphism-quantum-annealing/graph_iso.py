# This project implements the QUBO generation algorithm (Algorithm 1) of Experimental quantum annealing: case study involving the graph isomorphism problem by Kenneth M. Zick, Omar Shehab & Matthew French (doi:10.1038/srep11168)


# The algorithm takes two NetworkX graphs as input.

import networkx as nx
import logging
import time
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a log file handler
logFile = 'graph_iso_' + time.strftime("%d-%m-%Y") + '.log'

# handler = logging.FileHandler(logFile)
handler = RotatingFileHandler(logFile, mode='a', maxBytes=100*1024*1024, backupCount=100, encoding=None, delay=0)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


# This function returns the minimum number of QUBO varables
def get_restricted_qubo_matrix(num_nodes, graph1, graph2, logger):
   logger.info("Invoking get_restricted_qubo()...")

   # Creating the adjacency matrix dictionary from the input graphs
   adj1 = nx.adjacency_matrix(graph1)
   adj2 = nx.adjacency_matrix(graph2)
   
   logger.info("Graph 1:")
   logger.info(str(adj1))	  
   
   logger.info("Graph 2:")
   logger.info(str(adj2))

   # Construct a list of variables
   variables = []
   logger.info("Dictionary of variables  initialized.")

   for i in range(0, num_nodes):
      for j in range(0, num_nodes):
         if graph1.degree(i) == graph2.degree(j):
            variables.append((j, i))

   logger.info( "Dictionary of variables loaded.")
   logger.info(str(variables))
   
   num_of_restricted_variables = len(variables)
   
   # Creating an empty restricted QUBO matrix
   restricted_qubo_matrix = [[0 for x in range(num_of_restricted_variables)] for x in range(num_of_restricted_variables)]
   
   for i in range(0, len(variables)):
      for j in range(0, len(variables)):
         var1 = variables[i]
         var2 = variables[j]
         
         
      

   return

def run():
   num_of_nodes = 5
   edge_prob = 0.5
   graph1 = nx.erdos_renyi_graph(num_of_nodes, edge_prob)
   graph2 = nx.erdos_renyi_graph(num_of_nodes, edge_prob)
   variables = get_restricted_qubo_matrix (num_of_nodes, graph1, graph1, logger)
   return
