# Run the random.py program to generate random graphs.

from email.mime.text import MIMEText
import logging
from logging.handlers import RotatingFileHandler
import time
import randomg
import sys

# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a log file handler
logFile = 'logs/run_random_' + time.strftime("%d-%m-%Y") + '.log'

# handler = logging.FileHandler(logFile)
handler = RotatingFileHandler(logFile, mode='a', maxBytes=100*1024*1024, backupCount=100, encoding=None, delay=0)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


def run():
   # Configuration space
   nodes = 10000
   mem = 10000
   while nodes <= 50000:
      randomg.generate(nodes)
      nodes = nodes + 10000
      mem = get_free_memory()
      # Double check that we are not breaking BW
      if mem <= 10000:
         logger.info("The memory is less than " + str(mem) + " KB.")
         sys.exit()
   
         
      


def get_free_memory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
    return tmp
