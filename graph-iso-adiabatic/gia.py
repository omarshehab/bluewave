import matplotlib.pyplot as plt
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import time


from qutip import *
from scipy import *

# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a log file handler
logFile = 'logs/gia_' + time.strftime("%d-%m-%Y") + '.log'

# handler = logging.FileHandler(logFile)
handler = RotatingFileHandler(logFile, mode='a', maxBytes=100*1024*1024, backupCount=100, encoding=None, delay=0)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

N = 6   # number of spins
M = 20  # number of eigenenergies to plot

# array of spin energy splittings and coupling strengths (random values). 
h  = 1.0 * 2 * pi * (1 - 2 * rand(N))
logger.info("Type of h: " + str(type(h)))
logger.info(str(h))

Jz = 1.0 * 2 * pi * (1 - 2 * rand(N))
logger.info("Type of Jz: " + str(type(Jz)))
logger.info(str(Jz))

Jx = 1.0 * 2 * pi * (1 - 2 * rand(N))
logger.info("Type of Jx: " + str(type(Jx)))
logger.info(str(Jx))

Jy = 1.0 * 2 * pi * (1 - 2 * rand(N))
logger.info("Type of Jy: " + str(type(Jy)))
logger.info(str(Jy))


# increase taumax to get make the sweep more adiabatic
taumax = 5.0
taulist = np.linspace(0, taumax, 100)
logger.info("Type of taulist: " + str(type(taulist)))
logger.info(str(taulist))

# pre-allocate operators
si = qeye(2)
logger.info("Type of taulist: " + str(type(si)))
logger.info(str(si))

sx = sigmax()
logger.info("Type of taulist: " + str(type(sx)))
logger.info(str(sx))

sy = sigmay()
logger.info("Type of taulist: " + str(type(sy)))
logger.info(str(sy))

sz = sigmaz()
logger.info("Type of taulist: " + str(type(sz)))
logger.info(str(sz))


sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sy
    sy_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))

psi_list = [basis(2,0) for n in range(N)]
logger.info("Type of psi_list: " + str(type(psi_list)))
logger.info(str(psi_list))
 
psi0 = tensor(psi_list)
logger.info("Type of psi0: " + str(type(psi0)))
logger.info(str(psi0))

H0 = 0    
for n in range(N):
    H0 += - 0.5 * 2.5 * sz_list[n]
logger.info("Type of H0: " + str(type(H0)))
logger.info(str(H0))

# energy splitting terms
H1 = 0    
for n in range(N):
    H1 += - 0.5 * h[n] * sz_list[n]
logger.info("Energy splitting terms...")
logger.info("Type of H0: " + str(type(H1)))
logger.info(str(H1))


H1 = 0
for n in range(N-1):
    # interaction terms
    H1 += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
    H1 += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
    H1 += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]
logger.info("Interaction terms...")
logger.info("Type of H1: " + str(type(H1)))
logger.info(str(H1))

# the time-dependent hamiltonian in list-function format
args = {'t_max': max(taulist)}
logger.info("the time-dependent hamiltonian in list-function format...")
logger.info("Type of args: " + str(args))
logger.info(str(args))

h_t = [[H0, lambda t, args : (args['t_max']-t)/args['t_max']],
        [H1, lambda t, args : t/args['t_max']]]

logger.info("Type of args: " + str(h_t))
logger.info(str(h_t))



def main():
 # Start the timer
 experiment_start = time.time()
 logger.info("\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Experiment started --------------------------------------------------------------------------------")
 
 # Evolve the system, request the solver to call process_rho at each time step.

 mesolve(h_t, psi0, taulist, [], process_rho, args)

 #rc('font', family='serif')
 #rc('font', size='10')

 fig, axes = plt.subplots(2, 1, figsize=(12,10))

 #
 # plot the energy eigenvalues
 #

 # first draw thin lines outlining the energy spectrum
 for n in range(len(evals_mat[0,:])):
     ls,lw = ('b',1) if n == 0 else ('k', 0.25)
     axes[0].plot(taulist/max(taulist), evals_mat[:,n] / (2*pi), ls, lw=lw)

 # second, draw line that encode the occupation probability of each state in 
 # its linewidth. thicker line => high occupation probability.
 for idx in range(len(taulist)-1):
     for n in range(len(P_mat[0,:])):
         lw = 0.5 + 4*P_mat[idx,n]    
         if lw > 0.55:
            axes[0].plot(array([taulist[idx], taulist[idx+1]])/taumax, 
                         array([evals_mat[idx,n], evals_mat[idx+1,n]])/(2*pi), 
                         'r', linewidth=lw)    
        
 axes[0].set_xlabel(r'$\tau$')
 axes[0].set_ylabel('Eigenenergies')
 axes[0].set_title("Energyspectrum (%d lowest values) of a chain of %d spins.\n " % (M,N)
                 + "The occupation probabilities are encoded in the red line widths.")

 #
 # plot the occupation probabilities for the few lowest eigenstates
 #
 for n in range(len(P_mat[0,:])):
     if n == 0:
         axes[1].plot(taulist/max(taulist), 0 + P_mat[:,n], 'r', linewidth=2)
     else:
         axes[1].plot(taulist/max(taulist), 0 + P_mat[:,n])

 axes[1].set_xlabel(r'$\tau$')
 axes[1].set_ylabel('Occupation probability')
 axes[1].set_title("Occupation probability of the %d lowest " % M +
                   "eigenstates for a chain of %d spins" % N)
 axes[1].legend(("Ground state",));
 
 logger.info("Title set")
 logger.info("Saving the plot...")
 plot_file_name =  "gia_" + time.strftime("%d-%m-%Y")  + ".png"
 plt.savefig("plots/" + plot_file_name, dpi=1200)

 experiment_end = time.time()
 elapsed = experiment_end - experiment_start
 hours, rem = divmod(elapsed, 3600)
 minutes, seconds = divmod(rem, 60)
 log_string = "Time spent: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
 logger.info( log_string)
 log_string = "------|||||| END OF GIA EXPERIMENT ||||||------"
 logger.info( log_string)
 # me == the sender's email address
 # you == the recipient's email address

 # Send the message via our own SMTP server, but don't include the
 # envelope header.
 s = smtplib.SMTP('localhost')
 if email_report:
  s.sendmail("bluewave@chmpr.umbc.edu", ["shehab1@umbc.edu"], log_string)
 s.quit()




#
# callback function for each time-step
#
evals_mat = np.zeros((len(taulist),M))
P_mat = np.zeros((len(taulist),M))

idx = [0]

def process_rho(tau, psi):
    # evaluate the Hamiltonian with gradually switched on interaction 
    H = qobj_list_evaluate(h_t, tau, args)

    # find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0],:] = real(evals)

    # find the overlap between the eigenstates and psi 
    for n, eket in enumerate(ekets):
        P_mat[idx[0],n] = abs((eket.dag().data * psi.data)[0,0])**2

    idx[0] += 1


if __name__ == "__main__":
    main()
