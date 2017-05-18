import smtplib
import networkx as nx
from numpy import *
from email.mime.text import MIMEText
import logging
from logging.handlers import RotatingFileHandler
import time
import os

# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a log file handler
logFile = 'logs/random_' + time.strftime("%d-%m-%Y") + '.log'

# handler = logging.FileHandler(logFile)
handler = RotatingFileHandler(logFile, mode='a', maxBytes=100*1024*1024, backupCount=100, encoding=None, delay=0)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)



# This program generates very large Erdos-Renyi random graphs.
def generate(n):
   # Start the timer
   experiment_start = time.time()
   
   data_folder = "/data/s1/shehab1/graph-data/"

   
   nodes = n
   probability = 0.5
   email_report = True
   
   logger.info("\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Experiment started --------------------------------------------------------------------------------")

   
   log_string = "Generating graph with " + str(nodes) + "  nodes and edge probability " + str(probability)
   print log_string
   logger.info(log_string)

   api_call_start_time = time.time()

   graph = nx.erdos_renyi_graph(nodes, probability)
   print "Graph generated"

   api_call_end_time = time.time()
   api_call_elapsed_time = api_call_end_time - api_call_start_time
   api_call_hours, api_call_rem = divmod(api_call_elapsed_time, 3600)
   api_call_minutes, api_call_seconds = divmod(api_call_rem, 60)
   api_call_time_log_string = "API call time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(api_call_hours), int(api_call_minutes), api_call_seconds)

   print api_call_time_log_string
   logger.info(api_call_time_log_string)

   # me == the sender's email address
   # you == the recipient's email address

   # Send the message via our own SMTP server, but don't include the
   # envelope header.
   s = smtplib.SMTP('localhost')
   if email_report:
      s.sendmail("bluewave@chmpr.umbc.edu", ["shehab1@umbc.edu"], (log_string + "\n" + api_call_time_log_string))
   s.quit()

   file_write_start_time = time.time()

   fh=open(data_folder + str(nodes) + "-" + str(probability) + ".adjlist",'wb')
   nx.write_adjlist(graph, fh)
   print "Graph written to the file"

   file_write_end_time = time.time()
   file_write_elapsed_time = file_write_end_time - file_write_start_time
   file_write_hours, file_write_rem = divmod(file_write_elapsed_time, 3600)
   file_write_minutes, file_write_seconds = divmod(file_write_rem, 60)
   file_write_time_log_string = "File write time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(file_write_hours), int(file_write_minutes), file_write_seconds)
   print file_write_time_log_string
   logger.info(file_write_time_log_string)



   s = smtplib.SMTP('localhost')
   if email_report:
      s.sendmail("bluewave@chmpr.umbc.edu", ["shehab1@umbc.edu"], (log_string + "\n" + file_write_time_log_string))
   s.quit()
 
   file_zip_start_time = time.time()

