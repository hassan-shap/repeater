import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from EffP import *
from QLPDCgen import *
from PLmatrix_CSS import *

from ldpc.mod2 import rank,row_basis,inverse
from ldpc.codes import hamming_code
from bposd.css import css_code
from lifted_hgp import lifted_hgp
from ldpc import protograph as pt
from bposd.hgp import hgp
from ldpc.code_util import get_code_parameters

from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 12#multiprocessing.cpu_count()                                     

bdy = True ## boundary condition, true (obc), false(pbc)
repeat = 24
Nrep = 50 # number of iterations
Nl_list = [20,15,10,30]#np.arange(2,9)
# p_list = np.linspace(0.01,0.4,20)
p_list = np.linspace(0.01,0.4,20)

######## define quantum code here ########
a1=pt.array([
        [(0), (11), (7), (12)],
        [(1), (8), (1), (8)],
        [(11), (0), (4), (8)],
        [(6), (2), (4), (12)]])

print(a1)

H=a1.to_binary(lift_parameter=13)
n,k,d,_,_=get_code_parameters(H)
print(f"Code parameters: [{n},{k},{d}]")
qcode=lifted_hgp(lift_parameter=13,a=a1,b=a1)
#n = 418, k = 16, d ~ 20?
##########################################

##### code format & logical operators #####
Sx_mat = qcode.hx
Sz_mat = qcode.hz
Sx_mat = row_basis(Sx_mat)
Sz_mat = row_basis(Sz_mat)
qcode=css_code(hx=Sx_mat,hz=Sz_mat)
qcode.test()

lx=qcode.lx #x logical operators
lz=qcode.lz #z logical operators
print("X weight: ", np.sum(lx,axis=1))
print("Z weight: ", np.sum(lz,axis=1))
# lz[0,:] = (lz[1,:]+lz[0,:])%2

# print(qcode.compute_code_distance())
temp=inverse(lx@lz.T %2)
lx=temp@lx %2

print(lx.shape,lz.shape)
print( (lz@lx.T)% 2)
print(Sx_mat.shape,Sz_mat.shape)
print( np.linalg.norm((Sx_mat@Sz_mat.T) % 2))
print( np.linalg.norm((Sz_mat@lx.T) % 2))
print( np.linalg.norm((Sx_mat@lz.T) % 2))

print("X weight: ", np.sum(lx,axis=1))
print("Z weight: ", np.sum(lz,axis=1))

logical_tZ = lz
logical_tX = lx
N_logic = np.size(logical_tZ,0)
Nq_l = np.size(Sx_mat,1) # number of data qubits 
Ns_l = np.size(Sx_mat,0) # number of stabilizers 
##########################################


for i_L, Nl in enumerate(Nl_list):
    print("L= %d" % (Nl))
    
    N = Nl*(Nq_l+Ns_l) # number of data qubits
    Ns = Nl*Ns_l # number of stabilizers
    s_nodes = ["s%d" % s for s in np.arange(Ns)]

    B_orig_X = foliated_graph(Sx_mat,s_nodes, Nl, bdy)
    B_orig_Z = foliated_graph(Sz_mat,s_nodes, Nl, bdy)

    logical_in_X = np.zeros((N_logic,N))
    logical_in_Z = np.zeros((N_logic,N))
    for i_l in range(Nl):
        logical_in_X[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tX
        logical_in_Z[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tZ

    def runner(i_rep):
        tic = time.time()

        succ_prob_X = np.zeros((len(p_list),np.size(logical_tX,0)))
        succ_prob_Z = np.zeros((len(p_list),np.size(logical_tX,0)))
        succ_prob_word_X = np.zeros(len(p_list))
        succ_prob_word_Z = np.zeros(len(p_list))
        for i_p, p in enumerate(p_list):
            for i_r in range(Nrep):
                loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p)[:,0])
                succ_prob_X_val = succ_prob_css_q_resolved(B_orig_X, logical_in_X, s_nodes, loss_inds)
                succ_prob_X[i_p,:] += succ_prob_X_val
                succ_prob_word_X[i_p] += (np.sum(succ_prob_X_val)==N_logic)
                succ_prob_Z_val = succ_prob_css_q_resolved(B_orig_Z, logical_in_Z, s_nodes, loss_inds)
                succ_prob_Z[i_p,:] += succ_prob_Z_val
                succ_prob_word_Z[i_p] += (np.sum(succ_prob_Z_val)==N_logic)
                # if np.sum(succ_prob_Z_val)== N_logic and np.sum(succ_prob_X_val)== N_logic:
                    # succ_prob_word[i_p] += 1

        succ_prob_X /= (Nrep)
        succ_prob_Z /= (Nrep)
        succ_prob_word_X /= (Nrep)
        succ_prob_word_Z /= (Nrep)

        toc = time.time()
        print("finished L = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

        if bdy:
            fname = "data_418q_lp/" + "obc_qres_Nl_%d_i_%d.npz" % (Nl,i_rep)
        else:
            fname = "data_418q_lp/" + "qres_Nl_%d_i_%d.npz" % (Nl,i_rep)
            
        np.savez(fname, succ_prob_word_X=succ_prob_word_X, succ_prob_word_Z=succ_prob_word_Z, succ_prob_X=succ_prob_X, succ_prob_Z=succ_prob_Z, p_list=p_list, Nrep=Nrep)

        return 0

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))