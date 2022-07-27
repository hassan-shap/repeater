import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from EffP import *
from QLPDCgen import *
from PLmatrix_CSS import *

# generation of the code
l = 24
n = l*2
k = 6
r = 21
H_XZo = GB_gen(l,[2,8,15],[2,12,17])
H_XZ = H_standard(H_XZo, n, r)

# Monte Carlo simulation:
# have both X and Z calculation
# in layer stabilizer group
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

Nl_list = np.arange(2,3) # number of layers
p_list = np.linspace(0.01,0.4,20) # list of loss probability
Nrep = 30 # number of MC iterations
succ_prob_X = np.zeros((len(p_list),len(Nl_list)))
succ_prob_Z = np.zeros((len(p_list),len(Nl_list)))
succ_prob_tot = np.zeros((len(p_list),len(Nl_list)))

for i_L, Nl in enumerate(Nl_list):
    tic = time.time()
    N = Nl*(Nq_l+Ns_l) # number of data qubits
    Ns = Nl*Ns_l # number of stabilizers
    B_orig_X = nx.Graph()
    B_orig_X.add_nodes_from(np.arange(N))
    B_orig_Z = nx.Graph()
    B_orig_Z.add_nodes_from(np.arange(N))
    s_nodes = ["s%d" % s for s in np.arange(Ns)]
    B_orig_X.add_nodes_from(s_nodes)
    B_orig_Z.add_nodes_from(s_nodes)
    for row in range(Ns_l):
        qs_X = np.argwhere(Sx_mat[row,:]>0)[:,0]
        for i_l in range(Nl):
            B_orig_X.add_edges_from([("s%d" % ((i_l*Ns_l)+row), i_l*(Nq_l+Ns_l)+q) for q in qs_X])
        qs_Z = np.argwhere(Sz_mat[row,:]>0)[:,0]
        for i_l in range(Nl):
            B_orig_Z.add_edges_from([("s%d" % ((i_l*Ns_l)+row), i_l*(Nq_l+Ns_l)+q) for q in qs_Z])

    for i_l in range(Nl):
        B_orig_X.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), i_l*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
        B_orig_X.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), ((i_l-1)%Nl)*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
        B_orig_Z.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), i_l*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
        B_orig_Z.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), ((i_l-1)%Nl)*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])

    logical_in_X = np.zeros((np.size(logical_tX,0),N))
    logical_in_Z = np.zeros((np.size(logical_tX,0),N))
    for i_l in range(Nl):
        logical_in_X[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tX
        logical_in_Z[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tZ

    for i_p, p in enumerate(p_list):
        for i_rep in range(Nrep):
            loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p)[:,0])
            succ_prob_X[i_p,i_L] += succ_prob_css_calc(B_orig_X, logical_in_X, s_nodes, loss_inds)
            succ_prob_Z[i_p,i_L] += succ_prob_css_calc(B_orig_Z, logical_in_Z, s_nodes, loss_inds)

    toc = time.time()
    print("finished L = %d in %.1f secs" % (Nl,toc-tic))

succ_prob_X /= (Nrep * k)
succ_prob_Z /= (Nrep * k)
succ_prob_tot = np.multiply(succ_prob_X, succ_prob_Z)

print(succ_prob_X, succ_prob_Z, succ_prob_tot)