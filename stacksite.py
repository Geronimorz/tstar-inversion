


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:29:34 2020

@author: zhangyurong
"""
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import spline
sitespecdir='/Users/zhangyurong/tstar/plot/sitespec/'

# =============================================================================
# stalst = []
# fre = []
# freq = []
# spec = []
# x = [[]for i in range(1200)]
# y = [[]for i in range(1200)]
# allspec = [0.00] * 43
# num = [0] * 43
# 
# psitedir = '/Users/zhangyurong/tstar/data/sitespec/p/'
# ssitedir = '/Users/zhangyurong/tstar/data/sitespec/s/'
# 
# stafile = '/Users/zhangyurong/tstar/plot/stations.dat'
# stations = open('/Users/zhangyurong/tstar/data/sitespec/p/pstations','a')
# os.chdir('/Users/zhangyurong/tstar/1088seisP/result')
# 
# for line in open(stafile).readlines()[1:]:
#     sta = line.split()[2]
#     sta = sta[len(sta)-4:]
#     stalst.append(sta)
# # =============================================================================
# # for sta in stalst:  
# #     os.system('cat *S*%s.dat >> %s.dat'%(sta,sta))
# # =============================================================================
# # =============================================================================
# # stalst.remove('CIG2')
# # stalst.remove('INPS')
# # =============================================================================
# fre = np.linspace(0.333,14.333,43)
# for i in range(len(fre)):
#     fre[i]=format(fre[i],'.4f')
#     
# for sta in stalst:
#     print(sta)
#     sitefl=sta+'.dat'
#     for line in open(sitefl).readlines():
#         f = float(line.split()[0])
#         allspec[int(f/0.333)-1] += float(line.split()[1])
#         num[int(f/0.333)-1] += 1
#         freq.append(f)
#         spec.append(float(line.split()[1]))
#     if max(num)<10:
#         print(max(num))    
#         continue
#     n = 0
#     for i in range(len(freq)):
#         if i != 0 and int(abs(freq[i-1]-freq[i])/0.333) > 1:
#             n+=1
#         x[n].append(freq[i])
#         y[n].append(spec[i])
#     for i in range(len(allspec)):
#         allspec[i]=allspec[i]/num[i]
#         
# # =============================================================================
# #     sitefl = open(psitedir+sta+'siteP.dat','a')
# #     for i in range(len(fre)):
# #         sitefl.write('%f\n'%(allspec[i]))
# #     sitefl.close()
# #     stations.write('%s\n'%sta)
# # =============================================================================
#     
#     plt.clf()
# 
#     for i in range(n+1):
#         plt.plot(x[i],y[i],'lightgray')
#  
#     plt.plot(fre,allspec,c=(238/255,136/255,129/255), linewidth=2) 
#     plt.title('P wave Site Effect: %s %d traces'%(sta,n),fontsize=14)
#     plt.xlabel('Frequency(Hz)',fontsize=11)
#     plt.ylabel('Residual Spectrum(nm/s)',fontsize=11)
#     plt.savefig(sitespecdir+sta+'psite.pdf')
#     
#     freq = []
#     spec = []
#     x = [[]for i in range(1200)]
#     y = [[]for i in range(1200)]
#     allspec = [0.00] * 43
#     num = [0] * 43
#     
#         
#   
# stations.close()
# =============================================================================

import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import spline

stalst = []
fre = []
freq = []
spec = []
x = [[]for i in range(1200)]
y = [[]for i in range(1200)]
allspec = [0.00] * 73
num = [0] * 73

psitedir = '/Users/zhangyurong/tstar/data/sitespec/p/'
ssitedir = '/Users/zhangyurong/tstar/data/sitespec/s/'

stafile = '/Users/zhangyurong/tstar/plot/stations.dat'
# =============================================================================
# stations = open('/Users/zhangyurong/tstar/data/sitespec/p/pstations','a')
# =============================================================================
stations = open('/Users/zhangyurong/tstar/data/sitespec/s/sstations','a')
# =============================================================================
# os.chdir('/Users/zhangyurong/tstar/1088seisP/result')
# =============================================================================
os.chdir('/Users/zhangyurong/tstar/1088seisS/result')


for line in open(stafile).readlines()[1:]:
    sta = line.split()[2]
    sta = sta[len(sta)-4:]
    stalst.append(sta)
# =============================================================================
# for sta in stalst:  
#     os.system('cat *S*%s.dat >> %s.dat'%(sta,sta))
# =============================================================================
# =============================================================================
# stalst.remove('CIG2')
# stalst.remove('INPS')
# stalst.remove('CHUM')
# =============================================================================
# =============================================================================
# fre = np.linspace(0.333,14.333,43)
# =============================================================================
fre = np.linspace(0.2000,14.6000,73)
for i in range(len(fre)):
    fre[i]=format(fre[i],'.4f')
    
for sta in stalst:
    print(sta)
    sitefl=sta+'.dat'
    for line in open(sitefl).readlines():
        f = float(line.split()[0])
        allspec[(round((f/0.200)-1))] += float(line.split()[1])
        num[round((f/0.200)-1)] += 1
        freq.append(f)
        spec.append(float(line.split()[1]))
    if max(num)<10:
        print(max(num))    
        continue
    n = 0
    for i in range(len(freq)):
        if i != 0 and int(abs(freq[i-1]-freq[i])/0.200) > 1:
            n+=1
        x[n].append(freq[i])
        y[n].append(spec[i])
    for i in range(len(allspec)):
        allspec[i]=allspec[i]/num[i]
        
# =============================================================================
#     sitefl = open(psitedir+sta+'siteP.dat','a')
# =============================================================================
# =============================================================================
#     sitefl = open(ssitedir+sta+'siteS.dat','a')
#     for i in range(len(fre)):
#         sitefl.write('%f\n'%(allspec[i]))
#     sitefl.close()
# =============================================================================
# =============================================================================
#     stations.write('%s\n'%sta)
# =============================================================================
    
# =============================================================================
#     if sta == 'ABRA':
#         aa = []
#         for i in range(n+1):
#             aa.append(y[i][22])
#         aa.remove(8.44752121)
#         print(max(aa))
# =============================================================================
    
    plt.clf()

    for i in range(n+1):
        plt.plot(x[i],y[i],'lightgray')
        
    plt.plot(fre,allspec,c=(238/255,136/255,129/255), linewidth=2) 
    plt.title('S wave Site Effect: %s %d traces'%(sta,n),fontsize=14)
    plt.xlabel('Frequency(Hz)',fontsize=11)
    plt.ylabel('Residual Spectrum(nm/s)',fontsize=11)
    plt.savefig(sitespecdir+sta+'ssite.pdf')

    
    freq = []
    spec = []
    x = [[]for i in range(1200)]
    y = [[]for i in range(1200)]
    allspec = [0.00] * 73
    num = [0] * 73
    
        
  
stations.close()

# =============================================================================
# xnew = np.linspace(fre.min(),fre.max(),30) #300 represents number of points to make between T.min and T.max
#  
# power_smooth = smooth(fre,allspec)
# =============================================================================

# =============================================================================
# a = '/Users/zhangyurong/tstar/1088seisS/result/CALI.dat'
# f = []
# for line in open(a).readlines():
#     f.append(float(line.split()[0]))
# print(f.index(max(f)))
# print(max(f))
# =============================================================================
    