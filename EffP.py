import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from numpy.linalg import matrix_power
# from PLmatrix_CSS import *

def succ_prob_css_calc(B_orig, logicals_in, s_nodes, loss_inds):
    ######################################################
    ## inputs:
    ## B_orig [type: networkx]: stabilizer graph, two kinds of nodes: qubit 1...N and stabilizer s1...s_{Ns}
    ## logicals_in [type: list of numpy arrays]: logical operators in every row, columns act on qubits
    ## s_nodes [type: list]: list of stabilizer nodes s1...s_{Ns}
    ## loss_inds [type: numpy array]: index of erased qubits
    #####################
    ## output:
    ## succ_fail [type: binary value]: 0 (failure), 1(success)
    ######################################################
    B = B_orig.copy()
    logicals = list(np.copy(logicals_in))
    s_nodes_set = set(np.copy(s_nodes))
    N = np.shape(logicals_in)[1]

    Ns_remain = len(s_nodes_set) # number of stabilizer generators
    q_remain = list(set(B.nodes())-s_nodes_set) # number of qubits (anciall+data)
    node_list = list(s_nodes_set) + q_remain  # indices of all nodes in graph
    adj_mat_new = nx.to_numpy_array(B, nodelist = node_list) # adjaceny matrix of stabilizer graph
    Sx_mat = adj_mat_new[:Ns_remain,Ns_remain:] # stabilizer group matrix

    for i_q, q in enumerate(loss_inds):
        ## correct logical operators
        # if Ns_remain> 0:
        #     logicals = correct_logical(q,logicals, Sx_mat)
        # logicals = correct_logical(logicals, Sx_mat, q+Nq)
        if len(logicals) == 1:
            if logicals[0][q]>0:
                if Ns_remain> 0:
                    st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                    if len(st_ind)>0:
                        logicals[0] = (logicals[0]+Sx_mat[st_ind[0],:]) % 2
                else:
                    logicals.pop()
        else:
            for i_log in np.arange(len(logicals)-1,-1,-1):
                if logicals[i_log][q]>0:
                    if Ns_remain> 0:
                        st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                        if len(st_ind)>0:
                            logicals[i_log] = (logicals[i_log]+Sx_mat[st_ind[0],:]) % 2
                    else:
                        logicals.pop(i_log)            
        ## update stabilizer group
        ## first: update graph
        if q in B:
            B, s_nodes_set = modify_graph(q,B,s_nodes_set)
        ## second: update stabilizer group matrix
            Ns_remain = len(s_nodes_set)
            if Ns_remain> 0:
                q_remain = list(set(B.nodes())-s_nodes_set)
                node_list = list(s_nodes_set) + q_remain
                adj_mat_new = nx.to_numpy_matrix(B, nodelist = node_list)
                Sx_red = adj_mat_new[:Ns_remain,Ns_remain:]
                Sx_mat = np.zeros((Ns_remain,N))
                Sx_mat[:,q_remain] = Sx_red
            else:
                Sx_mat = []
                # break
        
        # logicals = correct_logical(logicals, Sx_mat, q+Nq)
        # q2 = q+Nq
        # if q2 in B:
        #     B, s_nodes_set = modify_graph(q2,B,s_nodes_set)
        # ## second: update stabilizer group matrix
        #     Ns_remain = len(s_nodes_set)
        #     if Ns_remain> 0:
        #         q_remain = list(set(B.nodes())-s_nodes_set)
        #         node_list = list(s_nodes_set) + q_remain
        #         adj_mat_new = nx.to_numpy_matrix(B, nodelist = node_list)
        #         Sx_red = adj_mat_new[:Ns_remain,Ns_remain:]
        #         Sx_mat = np.zeros((Ns_remain,N))
        #         Sx_mat[:,q_remain] = Sx_red
        #     else:
        #         Sx_mat = []
        #         break

    succ_fail = 0 # default value: failure
    # if Ns_remain > 0:
    #     return int(len(logicals)/2)
    # elif i_q == len(loss_inds)-1:
    #     return int(len(logicals)/2)
    num_qs = 0
    if len(logicals)>=1:
        for i_l in range(len(logicals)):
            if np.sum(logicals[i_l][loss_inds])==0:
                num_qs += 1 
            # print(logicals)
    return num_qs
    
def modify_graph(q,B,s_nodes_set):
    sq = [n for n in B.neighbors(q)]
    if len(sq)==1:
        B.remove_nodes_from(sq)
        s_nodes_set -= set(sq)
    elif len(sq)>1:
        double_edgs = []
        for i in range(len(sq)-1):
            n0 = set(B.neighbors(sq[i]))
            n1 = set(B.neighbors(sq[i+1]))
            rep_qs = n0.intersection(n1)
            q0 = list(set(n0)-rep_qs)
            q1 = list(set(n1)-rep_qs)
            double_edgs += [(sq[i], e) for e in q0]
            double_edgs += [(sq[i], e) for e in q1]
        G = nx.Graph()
        G.add_nodes_from(sq[:-1])
        G.add_edges_from(double_edgs)
        sq_remain = list(s_nodes_set-set(sq))
        for s in sq_remain:
            G.add_edges_from([(s, e) for e in B.neighbors(s)])
        B = G
        s_nodes_set -= {sq[-1]}
    return B, s_nodes_set


def correct_logical(q,logicals_in, Sx_mat):
    logicals = list(np.copy(logicals_in))
    if len(logicals) == 1:
        if logicals[0][q]>0:            
            st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
            if len(st_ind)>0:
                logicals[0] = (logicals[0]+Sx_mat[st_ind[0],:]) % 2
            else:
                logicals.pop()

    else:
        for i_log in np.arange(len(logicals)-1,-1,-1):
            if logicals[i_log][q]>0:            
                st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                if len(st_ind)>0:
                    logicals[i_log] = (logicals[i_log]+Sx_mat[st_ind[0],:]) % 2
                else:
                    logicals.pop(i_log)
    return logicals