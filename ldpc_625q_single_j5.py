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

repeat = 100
Nrep = 100 # number of iterations
Nl = 0
# p_list = np.linspace(0.01,0.4,20)
p_list = np.linspace(0.2,0.32,7)

#######
from bposd.hgp import hgp
from ldpc.code_util import get_code_parameters
H=np.array([\
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1,0,0,0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,0,1,0,0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,1,0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,0,0,0,1],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,1,0,0,0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,0,1,0,0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,0,0,1,0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,1,0,0,0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,0,1,0,0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,1,0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,1],
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,0,0,0,0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,0,0,0,0]])
print(sum(H[:]))
print(sum(np.transpose(H)[:]))
qcode=hgp(H,H,compute_distance=True)
qcode.test()

Sx_mat = qcode.hx
Sz_mat = qcode.hz

lx = qcode.lx
lz = qcode.lz
print("X weight: ", np.sum(lx,axis=1))
print("Z weight: ", np.sum(lz,axis=1))
######

print("Sx, Sz shapes:", Sx_mat.shape,Sz_mat.shape)
print("[Sx,Sz] = ", np.linalg.norm(Sx_mat@Sz_mat.T %2))

from ldpc.mod2 import rank,row_basis,inverse
# print(row_basis(Sx_mat).shape)
Sx_mat = row_basis(Sx_mat)
Sz_mat = row_basis(Sz_mat)
print(Sx_mat.shape,Sz_mat.shape)
# print(np.linalg.norm(Sx_mat@Sz_mat.T %2))

from ldpc.codes import hamming_code
from bposd.css import css_code

# print(qcode.compute_code_distance())
print("[lx,lz] = ", (lz@lx.T)% 2)

temp=inverse(lx@lz.T %2)
lx=temp@lx %2
    
print(lx.shape,lz.shape)
print("X weight: ", np.sum(lx,axis=1))
print("Z weight: ", np.sum(lz,axis=1))


print("[lx,lz] = ", (lz@lx.T)% 2)
# print(lz)
print("[Sx,Sz] = ", np.linalg.norm((Sx_mat@Sz_mat.T) % 2))
print("[Sz,lx] = ", np.linalg.norm((Sz_mat@lx.T) % 2))
print("[Sx,lz] = ", np.linalg.norm((Sx_mat@lz.T) % 2))

N_logic = np.size(lz,0)
N = np.size(Sx_mat,1) # number of data qubits 
Ns = np.size(Sx_mat,0) # number of stabilizers 
#############################
## construct stabilizer graph
B_orig_X = nx.Graph()
B_orig_X.add_nodes_from(np.arange(N))
B_orig_Z = nx.Graph()
B_orig_Z.add_nodes_from(np.arange(N))
s_nodes = ["s%d" % s for s in np.arange(Ns)]
B_orig_X.add_nodes_from(s_nodes)
B_orig_Z.add_nodes_from(s_nodes)

for row in range(Ns):
    qs_X = np.argwhere(Sx_mat[row,:]>0)[:,0]
    B_orig_X.add_edges_from([("s%d" % row, q) for q in qs_X])
    qs_Z = np.argwhere(Sz_mat[row,:]>0)[:,0]
    B_orig_Z.add_edges_from([("s%d" % row, q) for q in qs_Z])

def runner(i_rep):
    tic = time.time()

    succ_prob_X = np.zeros((len(p_list),np.size(lx,0)))
    succ_prob_Z = np.zeros((len(p_list),np.size(lx,0)))
    succ_prob_word_X = np.zeros(len(p_list))
    succ_prob_word_Z = np.zeros(len(p_list))
    for i_p, p in enumerate(p_list):
        for i_r in range(Nrep):
            loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p)[:,0])
            succ_prob_X_val = succ_prob_css_q_resolved(B_orig_X, lx, s_nodes, loss_inds)
            succ_prob_X[i_p,:] += succ_prob_X_val
            succ_prob_word_X[i_p] += (np.sum(succ_prob_X_val)==N_logic)
            succ_prob_Z_val = succ_prob_css_q_resolved(B_orig_Z, lz, s_nodes, loss_inds)
            succ_prob_Z[i_p,:] += succ_prob_Z_val
            succ_prob_word_Z[i_p] += (np.sum(succ_prob_Z_val)==N_logic)

    succ_prob_X /= (Nrep)
    succ_prob_Z /= (Nrep)
    succ_prob_word_X /= (Nrep)
    succ_prob_word_Z /= (Nrep)


    toc = time.time()
    print("finished L = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

    fname = "data_625q/" + "qres_Nl_%d_i_%d.npz" % (Nl,i_rep)

    np.savez(fname, succ_prob_word_X=succ_prob_word_X, succ_prob_word_Z=succ_prob_word_Z, succ_prob_X=succ_prob_X, succ_prob_Z=succ_prob_Z, p_list=p_list, Nrep=Nrep)

    return 0

results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))