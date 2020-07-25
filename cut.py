#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## PROGRAM TO PREPROCESS SAC FILES AND SAVE SEISMIC INFORMATION
## Written by Yurong Zhang, July. 2020

import os
import subprocess as sp
import numpy as np
import sys
sys.path.append('/Users/zhangyurong/tstar/tryprogram/haversine.py')
from haversine import distance_on_unit_sphere  as dist


Eventname = '/Users/zhangyurong/tstar/preprosessing/Eventname'
dirseismogram = '/Users/zhangyurong//tstar/preprosessing/processedSeismograms'
dbjadir = '/Users/zhangyurong/tstar/preprosessing/dbja'
stations = '/Users/zhangyurong/tstar/preprosessing/stations.dat'


for name in open(Eventname).readlines():
    name = name.strip()
    
    if not os.path.isdir(dbjadir):
        os.mkdir(dbjadir)
    dbjafl = dbjadir + '/dbja%s'%(name)
    dbja = open(dbjafl, 'a')
    dbja.write('sta, ptime, stime, delta, baz\n')
    
    os.chdir(dirseismogram+'/'+name)
    for line in open(stations).readlines()[1:]:
        sta = line.split()[2]
        cmd = 'saclst T0 T1 dist baz stla stlo evla evlo evdp f *%s*BHE*'%(sta)
##    If there is a file containing P and S wave arrivals, load them and don't read T0 and T1 here.
        p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        aa = p.communicate()
        a = aa.__str__()
        T0 = a.split()[1]
        T1 = a.split()[2]
        dis = a.split()[3]
        baz = a.split()[4]
        stla = a.split()[5]
        stlo = a.split()[6]
        evla = a.split()[7]
        evlo = a.split()[8]
        evdp = a.split()[9]
        evdp = evdp[:evdp.find('\\n')]
        evdp = float(evdp)/1000    ##UNITS: km
        
        if T1 == '-12345':
            T1 = '-1'
    
        dbja.write(sta + ' ' + T0 + ' ' + T1 + ' ' + dis + ' ' + baz + '\n')
    
        d = dist(float(stlo),float(stla),float(evlo),float(evla))
        way = np.sqrt(float(evdp)**2 + d**2)
        end = str( float(T0) + way/3 - way/5 + 200 )

        for chan in ['E','N','Z']:
            s = "cut " + str(float(T0) - 30.0) + " " + end + "\n"
            s += "r *" + sta + "*BH%s* \n"%(chan)
##            s += "decimate 2 \n"
            s += "rmean; rtr; taper \n"
            s += "ch t0 " + T0 + " t1 " + T1 + "\n"    
            s += "trans from evalresp fname RESPONSE_FILE_PATH to vel freq 0.005 0.01 10 20 \n"%(chan)    
            ## CHANGE RESPONSE_FILE_PATH TO 'RESPONSE FILE PATH'

            s += "bp co 0.05 15 n 4 p 2 \n"    
            s += "ch allt (0 - &1,T0&) iztype IT0 \n"
            s += "w over \n"
            s += "w alpha %s.HL%s.txt \n"%(sta, chan)
            s += "q \n"
            sp.Popen(['sac'], stdin=sp.PIPE).communicate(s.encode())
        
        
    
    dbja.close()
        
        
        
        
        
        
        
        
            
