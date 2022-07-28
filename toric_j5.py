import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from EffP import *
from QLPDCgen import *
from PLmatrix_CSS import *


from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 12#multiprocessing.cpu_count()                                     


bdy = True ## boundary condition, true (obc), false(pbc)
repeat = 24
Nrep = 10 # number of iterations
Nl_list = np.arange(2,9)
p_list = np.linspace(0.01,0.4,20)

L = 8
r1 = L
r2 = L
# in layer stabilizer group
Sx_mat = np.zeros((r1*r2,2*r1*r2),dtype=int)
for i_y in range(r2):
    for i_x in range(r1):
        Sx_mat[i_y*r1 + i_x, 2*(i_y*r1 + i_x)] = 1
        Sx_mat[i_y*r1 + i_x, 2*(i_y*r1 + i_x)+1] = 1
        Sx_mat[i_y*r1 + i_x, 2*(i_y*r1+(i_x-1)%r1 )] = 1
        Sx_mat[i_y*r1 + i_x, 2*(((i_y-1)%r2)*r1+i_x)+1] = 1
logicals_l = np.zeros((2,2*r1*r2))
logicals_l[0,1:2*r1:2] = np.ones(r1) 
logicals_l[1,0:2*r1*r2:2*r1] = np.ones(r2) 

Nq_l = np.size(Sx_mat,1) # number of data qubits per layer
Ns_l = np.size(Sx_mat,0) # number of stabilizers per layer

for i_L, Nl in enumerate(Nl_list):
    print("L= %d" % (Nl))
    
    N = Nl*(Nq_l+Ns_l) # number of data qubits
    Ns = Nl*Ns_l # number of stabilizers
    s_nodes = ["s%d" % s for s in np.arange(Ns)]

    B_orig = foliated_graph(Sx_mat,s_nodes, Nl, bdy)
    
    logicals = np.zeros((2,N))
    for i_l in range(Nl):
        logicals[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logicals_l

    def runner(i_rep):
        tic = time.time()
    
        succ_prob = np.zeros(len(p_list))
        for i_p, p in enumerate(p_list):
            for i_r in range(Nrep):
                loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p)[:,0])
                succ_prob[i_p] += succ_prob_css_calc(B_orig, logicals, s_nodes, loss_inds)
        
        succ_prob /= Nrep

        toc = time.time()
        print("finished L = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

        if bdy:
            fname = "data_toric/" + "obc_L%d_Nl_%d_i_%d.npz" % (L,Nl,i_rep)
        else:
            fname = "data_toric/" + "Nl_L%d_%d_i_%d.npz" % (L,Nl,i_rep)
            
        np.savez(fname, succ_prob=succ_prob, p_list=p_list, Nrep=Nrep)

        return 0

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))

