#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:48:28 2022

@author: daoniu
"""

# import
import phd_plot_style
import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, sqrt
from scipy.special import erf 
import os

# load style file
import matplotlib.font_manager
import matplotlib as mpl
import seaborn as sns
mpl.rcParams.update(mpl.rcParamsDefault)

pub_fig_style = phd_plot_style.phd_revtex_plots()

'''
# magic commands
%matplotlib inline
%config InlineBackend.print_figure_kwargs
%config InlineBackend.print_figure_kwargs={'bbox_inches':None, 'dpi': 200}
'''

# Colors
cBlues = sns.color_palette("Blues_r", n_colors=9)#[::3]
cOranges = sns.color_palette("Oranges_r", n_colors=7)#[::3]#[1:-1]
cGreens = sns.color_palette("Greens_r", n_colors=7)#[::3]#[1:-1]
cPurples = sns.color_palette("Purples_r", n_colors=7)#[::3]#[1:-1]
cGreys = sns.color_palette("PuRd_r", n_colors=7)#[::3]#[1:-1]
color_zip = [cPurples, cBlues, cOranges, cGreens, cGreys]

# define dimensions
fig_width  = pub_fig_style.revtex_sizes_active["page.columnwidth"] * pub_fig_style.tex_pt_to_inch
fig_height = pub_fig_style._get_revtex_rc()["figure.figsize"][1]

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def file_reader(fname_in,repeat):
    first = True
    for i_rep in range(repeat):
        fname = fname_in + "_i_%d.npz" % (i_rep)

        if os.path.exists(fname):
            npz_file = np.load(fname)
            succ_prob, p_list, Nrep = npz_file['succ_prob'], npz_file['p_list'], npz_file['Nrep']
            
            if first :
                first = False
                succ_prob_avg = np.zeros(len(p_list))
                Ntot = 0

            succ_prob_avg += succ_prob*Nrep
            Ntot += Nrep
    if not first:    
        return succ_prob_avg/Ntot, p_list, Ntot
    else:
        return 0,0,0
    
plt.figure()
p_r_list = [0.05,0.1]
legend_l = [r'$\eta_r=0.95$',r'$\eta_r=0.9$']
start_ind = [2,3]

for k_j in range(len(p_r_list)):
    
    repeat = 100
    p_r = p_r_list[k_j]
    Nl_list = np.arange(2,31,2)
    ind_p_list = np.arange(0,20,1)
    succ_prob_avg = np.zeros((len(ind_p_list),len(Nl_list)))
    
    plt.figure(1,figsize=(6,4))
    bdy = True
    for i_L, Nl in enumerate(Nl_list):
        f1 = "data_7q/" + "even_Nl_%d" % (Nl)
        # f1 = "old_data/data_7q/" + "obc_Nl_%d" % (Nl)
        succ_prob_even, p_list_even, Ntot_even = file_reader(f1,repeat) 
        succ_prob_repeater = succ_prob_even[np.argwhere(p_list_even==p_r)]
        # succ_prob_repeater = np.interp(p_r,p_list_even, succ_prob_even)
            
        # f2 = "old_data/data_7q/" + "obc_p_%.2f_Nl_%d" % (p_r,Nl)
        f2 = "data_7q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
        succ_prob_odd, p_list, Ntot = file_reader(f2,repeat) 
        succ_prob_avg[:,i_L] = succ_prob_odd[ind_p_list]*succ_prob_repeater
    
    slope_list = []
    slope_err_list = []
    succ_prob_avg_avg_array = succ_prob_avg.T
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
    for k in range(start_ind[k_j],succ_prob_avg_avg_array.shape[1]):
        # linear regression to fit the exponential decay:
        m,b = np.polyfit(Nl_fit_list[k], np.log(P_fit_list[k]), 1)
        slope_list.append(abs(m))
        # calculate the fitting slope error:
        yerr = m*np.array(Nl_fit_list[k])+b-np.array(np.log(P_fit_list[k]))
        N_len = len(Nl_fit_list[k])
        xavg_list = [sum(Nl_fit_list[k])/N_len]*N_len
        xerr = np.array(Nl_fit_list[k]) - np.array(xavg_list)
        slope_err = np.sqrt(1/N_len * sum(yerr**2) / sum(xerr**2) )
        slope_err_list.append(slope_err)
        
    A = np.log10((1-np.array(p_list[start_ind[k_j]:succ_prob_avg_avg_array.shape[1]]))/(1-p_r))
    L0_list=-50*A
    alpha_list=-0.2/np.log(10)*np.array(slope_list)/A
    alpha_err_list=-0.2/np.log(10)*np.array(slope_err_list)/A
    plt.errorbar(L0_list, alpha_list, yerr = alpha_err_list, marker='.',label=legend_l[k_j])

# p_r = 0 analytical solution:
f = lambda x: x**7 + 7*x**6*(1-x) + 7*3*x**5 * (1-x)**2 + 28 *x**4 * (1-x)**3 + 7*x**3 * (1-x)**4 
P_Pr0 = -10 / np.array(L0_list) * np.log10(f(10**(-0.02*np.array(L0_list))))
plt.plot(L0_list, P_Pr0, '.-', label = r'$\eta_r = 1$')

plt.legend(loc = "upper left",fontsize='x-large',frameon=False,ncol=1)
plt.xlabel(r'$L_0$ (km)')
plt.ylabel(r'$\alpha$ (dB/km)')
plt.title(r'$[[7,1,3]]$ code')
plt.savefig('Fig3_new_v1.pdf')
plt.show()