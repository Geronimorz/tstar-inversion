#!/usr/bin/env python2
# -*- coding: utf-8 -*-
## READ DATABASE, SET GLOBAL VARIABLES
import os
import subprocess as sp

def dbread(orid):
    global  figdir1, figdir2, figdir3, figdir4, figdir5, sacdir, csv, dbtime, dblat, dblon, dbdep, dbja, dbass
    figdir1='/Users/zhangyurong/tstar/workdir/specfig/plotseis'
    figdir2='/Users/zhangyurong/tstar/workdir/specfig/plotsnr'
    figdir3='/Users/zhangyurong/tstar/workdir/specfig/plotspec' 
    figdir4='/Users/zhangyurong/tstar/workdir/specfig/plottstar-fc'
    figdir5='/Users/zhangyurong/tstar/workdir/specfig/plotfall'
##    figdir6='/Users/zhangyurong/tstar/workdir/specfig/plotfft'
    sacdir='/Users/zhangyurong/tstar/data/processedSeismograms'
    csv = '/Users/zhangyurong/tstar/data/event.csv'
    dbjadir = '/Users/zhangyurong/tstar/data/dbja/'


    eventtime = orid.split('_')[1] + orid.split('_')[2] + orid.split('_')[3] + 'T' + orid.split('_')[4] + ':' + orid.split('_')[5] + ':' + orid.split('_')[6]
    for line in open(csv).readlines():
        if eventtime == line[0:17]:
            line = line.strip('\n')
            dbtime = line.split(', ')[0]
            dblat = line.split(', ')[1]
            dblon = line.split(', ')[2]
            dbdep = line.split(', ')[3]

    dbja = {}
    for line in open(dbjadir + 'dbja%s'%(orid)).readlines()[1:]:
        sta = line.split()[0]
        dbja[sta] = [0.00, 0,00, 0.00]
        dbja[sta][0] = float(line.split()[1])#ptime
        dbja[sta][1] = float(line.split()[2])#stime
        dbja[sta][2] = float(line.split()[3])#delta
        dbja[sta][3] = float(line.split()[4])#baz
    dbass = {}
    for line in open(dbjadir + 'dbja%s'%(orid)).readlines()[1:]:
        sta = line.split()[0]
        dbass[sta] = 360*float(line.split()[3])/40030.1736
        
    