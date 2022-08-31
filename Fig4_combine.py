#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:16:04 2022

@author: daoniu
"""

# load style file
import matplotlib.font_manager
import matplotlib as mpl
import seaborn as sns
mpl.rcParams.update(mpl.rcParamsDefault)
import phd_plot_style
pub_fig_style = phd_plot_style.phd_revtex_plots()

# magic commands
'''
%matplotlib inline
%config InlineBackend.print_figure_kwargs
%config InlineBackend.print_figure_kwargs={'bbox_inches':None, 'dpi': 200}
'''

# Colors
cBlues = sns.color_palette("Blues_r", n_colors=7)#[::3]
cOranges = sns.color_palette("Oranges_r", n_colors=7)#[::3]#[1:-1]
cGreens = sns.color_palette("Greens_r", n_colors=7)#[::3]#[1:-1]
cPurples = sns.color_palette("Purples_r", n_colors=7)#[::3]#[1:-1]
cGreys = sns.color_palette("PuRd_r", n_colors=7)#[::3]#[1:-1]
color_zip = [cPurples, cBlues, cOranges, cGreens, cGreys]

# define dimensions
fig_width  = pub_fig_style.revtex_sizes_active["page.columnwidth"] * pub_fig_style.tex_pt_to_inch
fig_height = pub_fig_style._get_revtex_rc()["figure.figsize"][1]
print(fig_width, fig_height)

import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, sqrt
from scipy.special import erf 
import os
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

def file_reader_ldpc(fname_in,Nq,repeat,even_odd):
    first = True
    for i_rep in range(repeat):
        fname = fname_in + "_i_%d.npz" % (i_rep)

        if os.path.exists(fname):
            npz_file = np.load(fname)
            if even_odd == "odd":
                succ_prob_word, succ_prob, p_list, Nrep = npz_file['succ_prob_word_X'], npz_file['succ_prob_X'],  npz_file['p_list'], npz_file['Nrep']
            else:
                succ_prob_word, succ_prob, p_list, Nrep = npz_file['succ_prob_word_Z'], npz_file['succ_prob_Z'],  npz_file['p_list'], npz_file['Nrep']

            if first :
                first = False
                succ_prob_avg = np.zeros((len(p_list),Nq))
                succ_prob_word_avg = np.zeros(len(p_list))
                Ntot = 0

            succ_prob_avg += succ_prob*Nrep
            succ_prob_word_avg += succ_prob_word*Nrep
            Ntot += Nrep
    if not first:    
        return succ_prob_avg/Ntot, succ_prob_word_avg/Ntot, p_list, Ntot
    else:
        return 0,0,0,0
    
plt.figure(1,figsize=(6,4))
p_r_list = [0.1,0.2]
fname_l = ["48q_pr10_p.npz","48q_pr20_p.npz"]
filename_l = ["48q_pr10_p.npz","48q_pr20_p.npz"]
legend_l = ['$p_r = 0.1$','$p_r = 0.2$']

for k_j in range(len(p_r_list)):
    p_r = p_r_list[k_j]
    p_repeater = 1 - p_r
    repeat = 200
    Nl_small = np.arange(2,9)
    Nl_large =  np.arange(12,21,4)
    Nl_list = np.concatenate((Nl_small,Nl_large))
    print(Nl_list)
    #Nl_list = np.arange(2,9,1)
    ind_p_list = np.arange(0,20,1)
    succ_prob_avg = np.zeros((len(ind_p_list),len(Nl_list)))
    
    
    bdy = True
    for i_L, Nl in enumerate(Nl_list):
        f1 = "data_48q/" + "even_Nl_%d" % (Nl)
        succ_prob_even,_, p_list_even, Ntot_even = file_reader_ldpc(f1,6,repeat,"even") 
        #succ_prob_repeater = succ_prob_even[np.argwhere(p_list_even==p_r)[0,:],:]
        #plt.plot(1-p_list_even,succ_prob_even,".", color="C%d" % (i_L), linewidth=1)#,label="N=%d, %d" % (Nl,Ntot))
            
        f2 = "data_48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
        succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd") 
        # succ_prob_avg[:,i_L] = (np.mean(succ_prob_odd,axis=1))[ind_p_list]*np.mean(succ_prob_repeater,axis=1)
        # succ_prob_avg[:,i_L] = np.mean( succ_prob_odd[ind_p_list,:]*succ_prob_repeater ,axis=1)
        succ_prob_avg[:,i_L] = np.mean(succ_prob_odd[ind_p_list,:] ,axis=1)
    
    succ_prob_avg_avg_array = succ_prob_avg.T
    succ_prob_tot = []
    Nl_list_large = []
    tot_layer = 30000
    for k in range(1,tot_layer):
        Nl_list_large.append((k+1))
    p_list0 = []
    P_fit_list = []
    Nl_fit_list = []
    for k in range(succ_prob_avg_avg_array.shape[1]):
        P_fit1 = []
        Nl_fit1 = []
        for j in range(succ_prob_avg_avg_array.shape[0]):
            if succ_prob_avg_avg_array[j,k] > 0:
                P_fit1.append(succ_prob_avg_avg_array[j,k])
                Nl_fit1.append(Nl_list[j])
        P_fit_list.append(P_fit1)
        Nl_fit_list.append(Nl_fit1)
        
    Nl_tot = len(list(Nl_list))
    for k in range(succ_prob_avg_avg_array.shape[1]):
        m,b = np.polyfit(Nl_fit_list[k], np.log(P_fit_list[k]), 1)
        if m < 0:
            succ_prob_tot.append(list(np.exp(m * np.array(Nl_list_large) + b)))
            p_list0.append(p_list[k])
    succ_prob_tot_array = np.array(succ_prob_tot)
    
    L_tot1 = 10**(np.arange(0,4,0.02))
    P_DT = 10**(-0.02*L_tot1)

### Nl_list_large Fig 4,5:
    fname = fname_l[k_j]
    succ_prob_avg_avg_list = list(succ_prob_tot_array.T)
    t_list = 1-np.array(p_list0)
    t_list = list(t_list)
    t_list.append(0)
    t_list.reverse()
    t_list.append(1)
    L_tot = 10**np.arange(0,4,0.02)

    each_dis_Peff_list = []
    P_optimal_list = []
    P_optimal_base = []
    Arrange_P_optimal_list = []
    cost_total = []
    for k in range(0,len(succ_prob_avg_avg_list)):
        succ_prob_avg_avg_list0 = list(succ_prob_avg_avg_list[k].copy())
        succ_prob_avg_avg_list0.append(0)
        succ_prob_avg_avg_list0.reverse()
        succ_prob_avg_avg_list0.append(1)
        P_eff_interpolate = interp1d(t_list, succ_prob_avg_avg_list0)
        #P_eff_interpolate = CubicSpline(t_list, succ_prob_avg_avg_list0, bc_type='natural')
        Num_repeater = Nl_list_large[k]
        p_keep = p_repeater*10**(-0.02*L_tot/Num_repeater)
        P_optimal_base.append(list(P_eff_interpolate(p_keep)))
        cost_total.append(list(Nl_list_large[k]/P_eff_interpolate(p_keep)))
    P_optimal_base = np.array(P_optimal_base)
    cost_total = np.array(cost_total)
    ### calculate cost:
    index_max = []
    P_max = []
    N_optimal_max = []
    P_ori_max = []
    index_Pori_max = []
    for k in range(len(L_tot)):
        cost_range = []
        P_max_range = []
        Nl_large_range = []
        
        '''
        for j_l in range(len(cost_total[:,k])):
            if P_optimal_base[j_l,k] > P_DT[k]:
                P_max_range.append(P_optimal_base[j_l,k])
                cost_range.append(cost_total[j_l,k])
                Nl_large_range.append(Nl_list_large[j_l])
        index_max.append(np.argmin(cost_range))
        P_max.append(P_max_range[index_max[k]])
        P_ori_max.append(max(P_optimal_base[:,k]))
        index_Pori_max.append(Nl_list_large[np.argmax(P_optimal_base[:,k])])
        N_optimal_max.append(Nl_large_range[index_max[k]])
        '''
        index_max.append(np.argmin(cost_total[:,k]))
        P_max.append(P_optimal_base[index_max[k],k])
        P_ori_max.append(max(P_optimal_base[:,k]))
        index_Pori_max.append(Nl_list_large[np.argmax(P_optimal_base[:,k])])
        N_optimal_max.append(Nl_list_large[index_max[k]])
    N_optimal_per10 = []
    for k in range(L_tot.shape[0]):
        N_optimal_per10.append(N_optimal_max[k] / L_tot[k] * 10)
    np.savez(fname, index_Pori_max = index_Pori_max, \
             P_ori_max = P_ori_max, N_optimal_per10 = N_optimal_per10,\
             P_max = P_max, index_max = index_max,\
             N_optimal_max=N_optimal_max, p_repeater = p_repeater,\
             L_tot = L_tot)

### Fig 4_1
fig = plt.figure()
ax1 = fig.add_subplot(111)
for k in range(len(filename_l)):
    fname = filename_l[k]
    if os.path.exists(fname):
        npz_file = np.load(fname)
        L_tot, N_optimal_per10 = npz_file['L_tot'], npz_file['N_optimal_per10']
        N_optimal_max = npz_file['N_optimal_max']
        index_Pori_max = npz_file['index_Pori_max']
        ax1.plot(L_tot*0.2, N_optimal_per10, '.-', label = legend_l[k])
ax1.set_xlabel("channel loss (dB)")
ax1.set_ylabel("Num of repeaters per 10km")
ax1.set_xlim(2, 2000)
#ax1.set_ylim(0,10)
ax1.set_xscale("log")
ax1.set_xticks([])
ax2 = ax1.twiny()
ax2.set_xlim(10, 10000)
ax2.set_xscale("log")
plt.savefig('Fig4_48q_1.pdf')
plt.show()

### Fig 4_2
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(L_tot1*0.2,P_DT,'--',color='k',label='direct transmission')
for k in range(len(filename_l)):
    fname = filename_l[k]
    if os.path.exists(fname):
        npz_file = np.load(fname)
        L_tot, P_max, P_ori_max= npz_file['L_tot'], npz_file['P_max'], npz_file['P_ori_max']
        ax1.plot(L_tot*0.2, P_max, '.-', label = legend_l[k])
        #ax1.plot(L_tot*0.2, P_ori_max, '.-', label = legend_l[k])
ax1.set_xlabel("channel loss (dB)")
ax1.set_ylabel("Effective P rate")
ax1.set_xlim(2, 2000)
ax1.set_xscale("log")
ax2 = ax1.twiny()
ax2.set_xlabel("total distance (km)")
ax2.set_xlim(10, 10000)
ax2.set_xscale("log")
ax1.legend(loc = "upper right",fontsize='large',frameon=False,ncol=1)
plt.savefig('Fig4_48q_2.pdf')
plt.show()