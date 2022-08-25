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
Nrep = 200 # number of iterations
Nl_list = np.arange(2,10)
p_list = np.linspace(0.01,0.4,20)

######## define quantum code here ########
l=63
n = 126
k = 28
r = 49
H_XZ = GB_gen(l,[1,14,16,22],[3,13,20,42])

##########
H_X = H_XZ[:, 0:n]
H_X = H_X[~np.all(H_X == 0, axis=1)]
H_Z = H_XZ[:, n:2*n]
H_Z = H_Z[~np.all(H_Z == 0, axis=1)]
print(H_X.shape, H_Z.shape)
n_row = np.size(H_X,axis=0)
H_Xb = []
H_Zb = []
for i_r in range(n_row):
    H_Xb.append(np.sum(H_X[i_r,:]* 2**np.arange(n-1,-1,-1)))
    H_Zb.append(np.sum(H_Z[i_r,:]* 2**np.arange(n-1,-1,-1)))

H_Xbc = H_Xb.copy()
H_Zbc = H_Zb.copy()
H_Xb1, RoX = gf2_rank(H_Xbc)
H_Zb1, RoZ = gf2_rank(H_Zbc)

print(len(H_Xb1))

H_X_new = Hb_to_H(n,r,H_Xb1)
H_Z_new = Hb_to_H(n,r,H_Zb1)
H_XZ = np.zeros((2*r,4*l), dtype = int)
H_XZ[0:r, 0:2*l] = H_X_new
H_XZ[r:2*r, 2*l:4*l] = H_Z_new

Sx_mat = H_XZ[:, 0:n]
Sx_mat = Sx_mat[~np.all(Sx_mat == 0, axis=1)]
Sz_mat = H_XZ[:, n:2*n]
Sz_mat = Sz_mat[~np.all(Sz_mat == 0, axis=1)]
Nq_l = np.size(Sx_mat,1) # number of data qubits per layer
Ns_l = np.size(Sx_mat,0) # number of stabilizers per layer
logicals_X = logical_from_Hparity(H_XZ, Nq_l, k, Ns_l)[0]
logicals_Z = logical_from_Hparity(H_XZ, Nq_l, k, Ns_l)[1]
logical_tX = logicals_X[:,0:Nq_l]
logical_tZ = logicals_Z[:,Nq_l:2*Nq_l]
##########################################

for i_L, Nl in enumerate(Nl_list):
    print("L= %d" % (Nl))
    
    N = Nl*(Nq_l+Ns_l) # number of data qubits
    Ns = Nl*Ns_l # number of stabilizers
    s_nodes = ["s%d" % s for s in np.arange(Ns)]

    B_orig_X = foliated_graph(Sx_mat,s_nodes, Nl, bdy)
    B_orig_Z = foliated_graph(Sz_mat,s_nodes, Nl, bdy)

    logical_in_X = np.zeros((np.size(logical_tX,0),N))
    logical_in_Z = np.zeros((np.size(logical_tZ,0),N))
    for i_l in range(Nl):
        logical_in_X[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tX
        logical_in_Z[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tZ

    def runner(i_rep):
        tic = time.time()

        succ_prob_X = np.zeros(len(p_list))
        succ_prob_Z = np.zeros(len(p_list))
        for i_p, p in enumerate(p_list):
            for i_r in range(Nrep):
                loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p)[:,0])
                succ_prob_X[i_p] += succ_prob_css_calc(B_orig_X, logical_in_X, s_nodes, loss_inds)
                succ_prob_Z[i_p] += succ_prob_css_calc(B_orig_Z, logical_in_Z, s_nodes, loss_inds)
        
        succ_prob_X /= (Nrep*k)
        succ_prob_Z /= (Nrep*k)

        toc = time.time()
        print("finished L = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

        if bdy:
            fname = "data_48q/" + "obc_Nl_%d_i_%d.npz" % (Nl,i_rep)
        else:
            fname = "data_48q/" + "Nl_%d_i_%d.npz" % (Nl,i_rep)
            
        np.savez(fname, succ_prob_X=succ_prob_X, succ_prob_Z=succ_prob_Z, p_list=p_list, Nrep=Nrep)

        return 0

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))

