#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 08:40:23 2022

@author: daoniu
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, sqrt
from scipy.special import erf 
import os

### Fig 2:
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

'''
repeat = 100
p_r= 0.1
Nl = 1
f2 = "data_fig2/7q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
succ_prob_odd, p_list, Ntot = file_reader(f2,repeat) 
plt.plot(p_list,1-succ_prob_odd,"v-",color="C0",label="[[7,1,3]],  $p_r$=%.1f, %d k" % (p_r,Ntot/1e3))
'''

repeat = 300
Nl = 1
#p_r_list = [0,0.1]
p_r_list = [0]
L = 6
Si = 7 #denote starting index in the data list for plotting

for i_p,p_r in enumerate(p_r_list):
    f2 = "data_fig2/48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd")
    
### plot part, adjust figure format:
    
plt.figure(1,figsize=(6,4))
f = lambda x: x**7 + 7*x**6*(1-x) + 7*3*x**5 * (1-x)**2 + 28 *x**4 * (1-x)**3 + 7*x**3 * (1-x)**4 
plt.plot(p_list[Si:],(1-f(1-p_list))[Si:],'-', color="C0")

marker_list = [".","v"]
for i_p,p_r in enumerate(p_r_list):
    f2 = "data_fig2/48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd") 
    plt.plot(p_list[Si:],(1-np.mean(succ_prob_odd,axis=1))[Si:],'-', color="C1")

    f2 = "data_toric/" +  "odd_L_%d_p_%.2f_Nl_%d" % (L,p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,2,repeat,"odd") 
    plt.plot(p_list[Si:],(1-np.mean(succ_prob_odd,axis=1))[Si:],'-', color="C2" )

### p_r = 0 part:
f = lambda x: x**7 + 7*x**6*(1-x) + 7*3*x**5 * (1-x)**2 + 28 *x**4 * (1-x)**3 + 7*x**3 * (1-x)**4 
plt.plot(p_list[Si:],(1-f(1-p_list))[Si:],'.',color="C0")

marker_list = [".","v"]
for i_p,p_r in enumerate(p_r_list):
    f2 = "data_fig2/48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd") 
    plt.plot(p_list[Si:],(1-np.mean(succ_prob_odd,axis=1))[Si:],'.', color="C1")

    f2 = "data_toric/" +  "odd_L_%d_p_%.2f_Nl_%d" % (L,p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,2,repeat,"odd") 
    plt.plot(p_list[Si:],(1-np.mean(succ_prob_odd,axis=1))[Si:],'.', color="C2")
    
### p_r = 0 errorbar:
#f = lambda x: x**7 + 7*x**6*(1-x) + 7*3*x**5 * (1-x)**2 + 28 *x**4 * (1-x)**3 + 7*x**3 * (1-x)**4 
#plt.errorbar(p_list[Si:],(1-f(1-p_list))[Si:],yerr = ,color="C0")

marker_list = [".","v"]
for i_p,p_r in enumerate(p_r_list):
    f2 = "data_fig2/48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd")
    p_fail = (1-np.mean(succ_prob_odd,axis=1))[Si:]
    plt.errorbar(p_list[Si:],p_fail,yerr = np.sqrt(p_fail*(1-p_fail)/Ntot), capsize=3,color="C1")

    f2 = "data_toric/" +  "odd_L_%d_p_%.2f_Nl_%d" % (L,p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,2,repeat,"odd")
    p_fail = (1-np.mean(succ_prob_odd,axis=1))[Si:]
    plt.errorbar(p_list[Si:],p_fail,yerr = np.sqrt(p_fail*(1-p_fail)/Ntot), capsize=3,color="C2")

### p_r = 0.1 part:
#plt.figure(1,figsize=(6,4))

repeat = 100
p_r= 0.1
Nl = 1
f2 = "data_fig2/7q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
succ_prob_odd, p_list, Ntot = file_reader(f2,repeat) 
plt.plot(p_list,1-succ_prob_odd,".-",color="C0",markerfacecolor='none',markersize=8)

p_r_list = [0.1]
repeat = 300
marker_list = [".","v"]
for i_p,p_r in enumerate(p_r_list):
    f2 = "data_fig2/48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd")
    plt.plot(p_list[Si:],(1-np.mean(succ_prob_odd,axis=1))[Si:],marker=marker_list[i_p], color="C1",markerfacecolor='none',markersize=8)

    f2 = "data_toric/" +  "odd_L_%d_p_%.2f_Nl_%d" % (L,p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,2,repeat,"odd") 
    plt.plot(p_list[Si:],(1-np.mean(succ_prob_odd,axis=1))[Si:],marker=marker_list[i_p], color="C2",markerfacecolor='none',markersize=8)

### p_r = 0.1  errorbar
f2 = "data_fig2/7q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
succ_prob_odd, p_list, Ntot = file_reader(f2,repeat)
p_fail = 1-succ_prob_odd
plt.errorbar(p_list,p_fail,yerr = np.sqrt(p_fail*(1-p_fail)/Ntot), capsize=3, color="C0",label="[[7,1,3]]")

p_r_list = [0.1]
repeat = 300
marker_list = [".","v"]
for i_p,p_r in enumerate(p_r_list):
    f2 = "data_fig2/48q/" + "odd_p_%.2f_Nl_%d" % (p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,6,repeat,"odd") 
    p_fail = (1-np.mean(succ_prob_odd,axis=1))[Si:]
    plt.errorbar(p_list[Si:],p_fail,yerr = np.sqrt(p_fail*(1-p_fail)/Ntot), capsize=3,color="C1",label=r"[[48,6,8]]")

    f2 = "data_toric/" +  "odd_L_%d_p_%.2f_Nl_%d" % (L,p_r,Nl)
    succ_prob_odd, _, p_list, Ntot = file_reader_ldpc(f2,2,repeat,"odd")
    p_fail = (1-np.mean(succ_prob_odd,axis=1))[Si:]
    plt.errorbar(p_list[Si:],p_fail,yerr = np.sqrt(p_fail*(1-p_fail)/Ntot), capsize=3,color="C2",label=r"6x6 Toric")

x_list = np.linspace(0.,0.3,1000)
plt.plot(x_list,x_list,"k--",linewidth=1,label='direct transmission')
    
plt.xlabel("loss rate")
plt.ylabel("Logical erasure prob.")
plt.yscale("log")

plt.ylim(1e-6,1e0)
plt.xlim(p_list[Si],0.3)
#plt.legend(frameon=False, fontsize='large', ncol=1,bbox_to_anchor=(1.04, 1), loc="upper left")
plt.legend(frameon=False, fontsize='large', ncol=1, loc="lower right")
plt.tight_layout()
plt.savefig("Fig2_v5.pdf")