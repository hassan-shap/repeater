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
repeat = 100
Nrep = 100 # number of iterations
Nl_list = np.arange(1,2)
p_list = [0.05,0.1,0.15,0.2]

######## define quantum code here ########
L = 6
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
logical_tX = np.zeros((2,2*r1*r2))
logical_tX[0,1:2*r1:2] = np.ones(r1) 
logical_tX[1,0:2*r1*r2:2*r1] = np.ones(r2) 

Nq_l = np.size(Sx_mat,1) # number of data qubits per layer
Ns_l = np.size(Sx_mat,0) # number of stabilizers per layer
N_logic = np.size(logical_tX,0) 
##########################################

for i_L, Nl in enumerate(Nl_list):
    print("L= %d" % (Nl))
    
    N = Nl*(Nq_l+Ns_l)+Ns_l # number of data qubits
    Ns = Nl*Ns_l # number of stabilizers
    s_nodes = ["s%d" % s for s in np.arange(Ns)]

    B_orig_Z = foliated_graph(Sx_mat,s_nodes, Nl, bdy, "even")

    logical_in_Z = np.zeros((N_logic,N))
    for i_l in range(Nl):
        logical_in_Z[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tX

    def runner(i_rep):
        tic = time.time()

        succ_prob_Z = np.zeros((len(p_list),np.size(logical_tX,0)))
        succ_prob_word_Z = np.zeros(len(p_list))
        for i_p, p_r in enumerate(p_list):
            p_stab = 1-(1-p_r)**0.5
            for i_r in range(Nrep):
                loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p_stab)[:,0])
                succ_prob_Z_val = succ_prob_css_q_resolved(B_orig_Z, logical_in_Z, s_nodes, loss_inds)
                succ_prob_Z[i_p,:] += succ_prob_Z_val
                succ_prob_word_Z[i_p] += (np.sum(succ_prob_Z_val)==N_logic)
               
        succ_prob_Z /= (Nrep)
        succ_prob_word_Z /= (Nrep)


        toc = time.time()
        print("finished Nl = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

        if bdy:
            fname = "data_toric/" + "even_L_%d_Nl_%d_i_%d.npz" % (L,Nl,i_rep)
        else:
            assert 0    
        np.savez(fname,  succ_prob_word_Z=succ_prob_word_Z, succ_prob_Z=succ_prob_Z, p_list=p_list, Nrep=Nrep)

        return 0

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))