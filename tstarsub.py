#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import subprocess
import numpy as np
import globaldb as g
from scipy.signal import *
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size']=8
mpl.rcParams['axes.formatter.limits']=[-2,2]
#import pymutt
import sys
sys.path.append('/Users/zhangyurong/tstar/program!/mtspec-0.3.2/build/lib.macosx-10.7-x86_64-3.7/mtspec')
import multitaper as mtm
import seissign as seis
##import forward_modeling as fm
## Subrountines for inverting t*
## Written by S. Wei, Nov. 2013
## Edited by S. Wei for t*(S), Feb. 2014
## Edited by Yurong Zhang, Jul. 2020
## readseismo:  extract raw data from antelope database
## getdbinfo:   get informatio of station and event
## fixwin:      window seismic data with a fixed length
## dospec:      calculate spectra of whole data, P/S signal, and P/S noise

def readseismo(pretime,dt,orid,sta,chan):
## Read seismogram from extsac_tstar.py 
## USAGE: (dd,time,flag) = readseismo(pretime,dt,orid,sta,chan)
## INPUT:  (pretime)  ==> Seconds before arrival to subset
##         (dt)       ==> Scalar: 1/(samplerate of data)
##         (subdb)    ==> Subset of database
##         (orid)     ==> Origin ID of event
##         (sta)      ==> Station name
##         (chan)     ==> Channel names (i.e. ['BHE','BHN','BHZ'])
## OUTPUT: (dd)       ==> (3Xn) numpy.ndarray: [(E_CHAN),(N_CHAN),(Z_CHAN)]
##         (time)     ==> (1Xn) numpy.ndarray: (Relative Time)
##         (flag)     ==> If reading successful
    
    for i in range(len(chan)):
        sacfl=g.sacdir+'/%s/%s.%s.txt'%(orid, sta, chan[i])
##        print(sacfl)
        if not os.path.isfile(sacfl):
            print('ERROR: %s does not exist' % sacfl)
            dd=np.array(range(18000))
            tt=np.array(range(18000))
            flag=False
            return dd,tt,flag
        ddchan=np.fromstring(("".join(open(sacfl).readlines()[30:])),sep=' ')
        nn=ddchan.size
        if i == 0:
            nmin = nn
        else:
            if nn < nmin:
                nmin = nn
        
    for ichan in range(len(chan)):
        sacfl=g.sacdir+'/%s/%s.%s.txt'%(orid, sta, chan[ichan])
##        print(sacfl)
        if not os.path.isfile(sacfl):
            print('ERROR: %s does not exist' % sacfl)
            dd=np.array(range(18000))
            tt=np.array(range(18000))
            flag=False
            return dd,tt,flag
        ddchan=np.fromstring(("".join(open(sacfl).readlines()[30:])),sep=' ')
##        ddchan=ddchan/1e9
##        ddchan=np.genfromtxt(sacfl,skiprows=30,invalid_raise=False)
        ddchan = ddchan[:nmin]
##        ddchan=ddchan.reshape(1,nn)
        if ichan==0:
            dd=ddchan
        else:
            dd=np.vstack((dd,ddchan))
    flag=True
    tt=-pretime+dt*np.array(range(nmin))
    return dd,tt,flag


def getdbinfo(sta,orid,ORIG):
## Get desired information about given database
## USAGE: (ARRIV,Ptt,Stt) = getdbinfo(sta,orid,ORIG)
## INPUT: (sta)  = station name
##        (orid) = origin id (orid)
##        (ORIG) = Origin Disctionary with Keys:
##            [orid] = origin id (orid)
##            [time] = origin time (unix seconds)
##            [lat]  = hypocenter latitude
##            [lon]  = hypocenter longitude
##            [dep]  = hypocenter depth
##            [ml]   = local magnitude
## OUTPUT: (ARRIV) = Arrival Dictionary with Keys:
##            [sta]   = station name
##            [ptime] = P arrival time (unix seconds)--to origin
##            [parid] = P arrival id (arid)
##            [pchan] = channel that P was picked on
##            [stime] = S arrival time (unix seconds)--to origin
##            [sarid] = S arrival id (arid)
##            [schan] = channel that S was picked on
##            [delta] = distance in degrees from sta to origin
##            [baz]   = station to event azimuth
##         (Ptt) = P travel time
##         (Stt) = S travel time
##       (SDATA) = If S arrival exists
    
    ARRIV={}
    ARRIV['sta']=sta
    g.dbread(orid)
    ## GET P ARRIVEL INFORMATION
    (ARRIV['ptime'], ARRIV['delta'], ARRIV['baz'], ARRIV['pchan'])=(g.dbja[sta][0], 360*g.dbja[sta][2]/40030.1736, g.dbja[sta][3], 'BHZ')
    Ptt=ARRIV['ptime']
    ## GET S ARRIVEL INFORMATION
    if g.dbja[sta][1] == -1:
        ARRIV['stime']=111
        Stt=Ptt+1
        SDATA=False
    else:
        ARRIV['stime'] = g.dbja[sta][1]
        Stt=ARRIV['stime']
        SDATA=True

    return ARRIV,Ptt,Stt,SDATA


def fixwin(dd,tt,dt,chan,ARRIV,prewin,WLP,WLS,SDATA,doplot,orid,sta,PStooclose):
## Window seismic data
## USAGE: (p_dd,s_dd,pn_dd,sn_dd,NOS2)
##            =tstarsub.fixwin(dd,tt,dt,ARRIV[ii+1],prewin,WLP,WLS,NOS,doplot,orid,sta)
## INPUT: (dd)      ==> (3Xn) numpy.ndarray: [(E_CHAN),(N_CHAN),(Z_CHAN)]
##        (tt)      ==> (1Xn) numpy.ndarray: (Relative Time)
##        (dt)      ==> Scalar: 1/(samplerate of data)
##        (chan)    ==> List of channal names
##        (ARRIV)   ==> Arrival Dictionary with Keys (see getdbinfo)
##        (prewin)  ==> Seconds before P[0]/S[1] arrival for windowing
##        (WLP)     ==> Window lenght for P in seconds
##        (WLS)     ==> Window lenght for S in seconds
##        (SDATA)   ==> Existence of S arrival
##        (doplot)  ==> Bool variable for plotting spectrum
##        (orid)    ==> origin id
##        (sta)     ==> station name
## OUTPUT: (p_dd)   ==> Windowed P data, starting prewin seconds before P arrival
##         (s_dd)   ==> Windowed S data
##         (pn_dd)  ==> P noise data, WL length ending prewin seconds before P arrival
##         (sn_dd)  ==> S noise data
##         (pd_dd)  ==> P coda data, right before S arrivel

    gaps=0.5      ## seconds between S noise and signal
    ## WINDOWING P
    pch=chan.index(ARRIV['pchan'])
    pind=np.all([(tt>=-1*prewin[0]),(tt<=(WLP-prewin[0]))],axis=0)
    p_dd=detrend(dd[pch][pind]-np.mean(dd[pch][pind]))
    pnind=np.all([(tt<=-1*prewin[0]),(tt>=(-WLP-prewin[0]))],axis=0)
    pn_dd=detrend(dd[pch][pnind]-np.mean(dd[pch][pnind]))
    ## MAKING SURE P WAVE AND NOISE HAVE SAME SIZE
    if p_dd.size>pn_dd.size:
        p_dd=p_dd[0:(pn_dd.size)]
    elif p_dd.size<pn_dd.size:
        pn_dd=s_dd[0:(p_dd.size)]
    if doplot:
        scalefac=1e3
        plt.figure(3)
        plt.clf()
        plt.subplot(2,1,2)
        plt.xlabel(ARRIV['pchan']+' (seconds)')
        plt.ylabel('Velocity Amplitude (nm/s)')
        plt.title('Station: %s' % sta)
#        plt.plot(tt,dd[pch])
        tmpdd=dd[pch]
        b, a = butter(4, 0.01, btype='highpass')
        filtmpdd = filtfilt(b, a, tmpdd)
        plt.plot(tt,filtmpdd)
#        plt.xlim([-WLP-prewin[0]-3,WLP-prewin[0]+3])
        plt.xlim([-10,10])
##        plt.ylim([-20e2,20e2])
        plt.ylim([-3e5,3e5])
#        plt.ylim([np.floor(min(filtmpdd)/scalefac)*scalefac,np.ceil(max(filtmpdd)/scalefac)*scalefac])
        plt.axvline(-1*prewin[0],color='g')
        plt.axvline(WLP-prewin[0],color='g')
        plt.axvline(-WLP-prewin[0],color='g')
        plt.text(-WLP-prewin[0],min(p_dd),'Noise', fontsize = 15)
        plt.text(-1*prewin[0],min(p_dd),'P wave',fontsize = 15)
        plt.savefig(g.figdir1+'/%s_%s_Pwindata.pdf' % (orid,sta))
    ## WINDOWING S ON BOTH HORIZONTAL CHANNELS
    if SDATA:
        sminusp=ARRIV['stime']-ARRIV['ptime']
        if sminusp<(WLS+WLP+prewin[0]-prewin[1]+gaps):
            PStooclose += 1
            print('P & S arrivels are too close - proceeding as if no S pick')
            SDATA=False
        if round(sminusp/0.025,5) == int(sminusp/0.025):
            sminusp += 0.001
    if SDATA:
##        sch=chan.index(ARRIV['schan'])
        ## HORIZONTAL CHANNEL 1
        sch=0
        sind=np.all([(tt>=(sminusp-prewin[1])),(tt<=(sminusp+WLS-prewin[1]))],axis=0)
        s_dd1=detrend(dd[sch][sind]-np.mean(dd[sch][sind]))
######## Noise defined as gaps s before S arrival
        snind=np.all([(tt<=(sminusp-prewin[1]-gaps)),(tt>=(sminusp-WLS-prewin[1]-gaps))],axis=0)
        sn_dd1=detrend(dd[sch][snind]-np.mean(dd[sch][snind]))
########## Noise defined as right before P arrival
##        snind=np.all([(tt<=-1*prewin[1]),(tt>=(-WLS-prewin[1]))],axis=0)
##        sn_dd=detrend(dd[sch][snind]-np.mean(dd[sch][snind]))
######## P coda defined as right before S arrival
        pcind=np.all([(tt<=(sminusp-prewin[1])),(tt>=(sminusp-WLS-prewin[1]))],axis=0)
        pc_dd1=detrend(dd[sch][pcind]-np.mean(dd[sch][pcind]))
        ## MAKING SURE S WAVE, NOISE AND P CODA HAVE SAME SIZE
        minlen=min(s_dd1.size,sn_dd1.size,pc_dd1.size)
        s_dd1=s_dd1[0:minlen]
        sn_dd1=sn_dd1[0:minlen]
        pc_dd1=pc_dd1[0:minlen]
        ## HORIZONTAL CHANNEL 2
        sch=1
        sind=np.all([(tt>=(sminusp-prewin[1])),(tt<=(sminusp+WLS-prewin[1]))],axis=0)
        s_dd2=detrend(dd[sch][sind]-np.mean(dd[sch][sind]))
######## Noise defined as right before S arrival
        snind=np.all([(tt<=(sminusp-prewin[1]-gaps)),(tt>=(sminusp-WLS-prewin[1]-gaps))],axis=0)
        sn_dd2=detrend(dd[sch][snind]-np.mean(dd[sch][snind]))
########## Noise defined as right before P arrival
##        snind=np.all([(tt<=-1*prewin[1]),(tt>=(-WLS-prewin[1]))],axis=0)
##        sn_dd=detrend(dd[sch][snind]-np.mean(dd[sch][snind]))
######## P coda defined as right before S arrival
        pcind=np.all([(tt<=(sminusp-prewin[1])),(tt>=(sminusp-WLS-prewin[1]))],axis=0)
        pc_dd2=detrend(dd[sch][pcind]-np.mean(dd[sch][pcind]))
        ## MAKING SURE S WAVE, NOISE AND P CODA HAVE SAME SIZE
        minlen=min(s_dd2.size,sn_dd2.size,pc_dd2.size)
        s_dd2=s_dd2[0:minlen]
        sn_dd2=sn_dd2[0:minlen]
        pc_dd2=pc_dd2[0:minlen]
        if doplot:
            scalefac=2e3
            plt.figure(4)
            plt.clf()
            plt.subplot(2,1,1)
#            plt.xlabel(ARRIV['schan']+' (seconds)')
            plt.xlabel(chan[0]+' (seconds)')
            plt.ylabel('Velocity Amplitude (nm/s)')
            plt.title('Station: %s' % sta)
#            plt.plot(tt,dd[0])
            tmpdd=dd[0]
            b, a = butter(4, 0.01, btype='highpass')
            filtmpdd = filtfilt(b, a, tmpdd)
            plt.plot(tt,filtmpdd)
########## P coda defined as right before S arrival
##            plt.axvline(sminusp-WLS-prewin[1],color='g')
##            plt.text(sminusp-WLP-prewin[1],min(s_dd),'P coda')
######## Noise defined as right before S arrival
##            plt.xlim([sminusp-WLS-prewin[1]-10,sminusp+WLS-prewin[1]+10])
##            plt.plot([sminusp-prewin[1],sminusp-prewin[1]],[min(dd[sch]),max(dd[sch])],'g')
##            plt.plot([sminusp+WLS-prewin[1],sminusp+WLS-prewin[1]],[min(dd[sch]),max(dd[sch])],'g')
##            plt.plot([sminusp-WLS-prewin[1],sminusp-WLS-prewin[1]],[min(dd[sch]),max(dd[sch])],'g')
#            plt.xlim([-WLP-prewin[1]-5,sminusp+WLS-prewin[1]+5])
#            plt.xlim([sminusp-WLS-gaps-prewin[1]-3,sminusp+WLS-prewin[1]+3])
            plt.xlim([sminusp-10,sminusp+10])
            plt.ylim([-1e6,1e6]);
#            plt.ylim([np.floor(min(filtmpdd)/scalefac)*scalefac,np.ceil(max(filtmpdd)/scalefac)*scalefac])
            plt.axvline(sminusp-prewin[1],color='g')
            plt.axvline(sminusp+WLS-prewin[1],color='g')
            plt.axvline(sminusp-prewin[1]-gaps,color='g')
            plt.axvline(sminusp-WLS-prewin[1]-gaps,color='g')
            plt.axvline(-1*prewin[1],color='g')
            plt.text(-prewin[1],min(s_dd1),'P arrival')
            plt.text(sminusp-WLS-prewin[1]-gaps,min(s_dd1),'Noise', fontsize = 15)
            plt.text(sminusp-prewin[1],min(s_dd1),'S wave', fontsize = 15)
            
########## Noise defined as right before P arrival
##            plt.xlim([-WLP-prewin[1]-10,sminusp+WLS-prewin[1]+10])
##            plt.axvline(sminusp-prewin[1],color='g')
##            plt.axvline(sminusp+WLS-prewin[1],color='g')
##            plt.axvline(-1*prewin[1],color='g')
##            plt.axvline(-WLS-prewin[1],color='g')
##            plt.text(-WLP-prewin[1],min(s_dd),'Noise')
##            plt.text(-prewin[1],min(s_dd),'P arrival')
##            plt.text(sminusp-prewin[1],min(s_dd),'S wave')
            plt.subplot(2,1,2)
#            plt.xlabel(ARRIV['schan']+' (seconds)')
            plt.xlabel(chan[1]+' (seconds)')
            plt.ylabel('Velocity Amplitude (nm/s)')
#            plt.plot(tt,dd[1])
            tmpdd=dd[1]
            b, a = butter(4, 0.01, btype='highpass')
            filtmpdd = filtfilt(b, a, tmpdd)
            plt.plot(tt,filtmpdd)
##            plt.xlim([-WLP-prewin[1]-10,sminusp+WLS-prewin[1]+10])
#            plt.xlim([sminusp-WLS-gaps-prewin[1]-3,sminusp+WLS-prewin[1]+3])
            plt.xlim([sminusp-10,sminusp+10])
            plt.ylim([-1e6,1e6]);
#            plt.ylim([np.floor(min(filtmpdd)/scalefac)*scalefac,np.ceil(max(filtmpdd)/scalefac)*scalefac])
            plt.axvline(sminusp-prewin[1],color='g')
            plt.axvline(sminusp+WLS-prewin[1],color='g')
            plt.axvline(sminusp-prewin[1]-gaps,color='g')
            plt.axvline(sminusp-WLS-prewin[1]-gaps,color='g')
#            plt.axvline(-1*prewin[1],color='g')
#            plt.text(-prewin[1],min(s_dd2),'P arrival')
            plt.text(sminusp-WLS-prewin[1]-gaps,min(s_dd2),'Noise', fontsize = 15)
            plt.text(sminusp-prewin[1],min(s_dd2),'S wave', fontsize = 15)
            plt.savefig(g.figdir1+'/%s_%s_Swindata.pdf' % (orid,sta))
    else:
        s_dd1=p_dd
        sn_dd1=pn_dd
        pc_dd1=p_dd
        s_dd2=p_dd
        sn_dd2=pn_dd
        pc_dd2=p_dd
    return p_dd,pn_dd,s_dd1,sn_dd1,pc_dd1,s_dd2,sn_dd2,pc_dd2,SDATA,PStooclose


def longseg(snr,snrcrtpara,freq,minf=0.35,maxf=15):
## FIND THE LONGEST SEGMENT OF SPECTRA WITH SNR > SNRCRT
## USAGE: (begind,endind,frmin,frmax,frange)=longseg(snr,snrcrtp,freq)
## INPUT: snr        = SIGNAL-TO-NOISE RATIO
##        snrcrtpara = [MINIMUM SIGNAL-TO-NOISE, MINIMUM LENGTH OF SEGMENT]
##        freq       = FREQUENCY
## OUTPUT: begind = INDEX OF THE BEGIN OF THE LONGEST SEGMENT
##         endind = INDEX OF THE END OF THE LONGEST SEGMENT
##         frmin  = MINIMUM FREQUENCY OF THE LONGEST SEGMENT
##         frmax  = MAXIMUM FREQUENCY OF THE LONGEST SEGMENT
##         frange = frmax - frmin
    ## TAKE SPECTRUM < maxf (15 Hz) and > minf (0.1 Hz)
##    print('Find longest frequency band')
    lenspec=len([ifreq for ifreq in freq if (ifreq<maxf and ifreq>=minf)])
    ind1=int(min(np.nonzero(freq>=minf)[0]))
    ind2=int(max(np.nonzero(freq<maxf)[0]))
    w=0
    m=[]
    bindex=[]
    eindex=[]
    snrcrt=snrcrtpara[0]
    for kk in range(ind1+1,lenspec):
        if snr[kk]<snrcrt and snr[kk-1]>=snrcrt and kk==1:        # only first > crt
            w=1
            m.append(w)
            bindex.append(kk-w)
            eindex.append(kk-1)
            w=0
        elif snr[kk]>=snrcrt and snr[kk-1]>=snrcrt and kk==1:     # at first and continuously > crt
            w=w+2
        elif snr[kk]>=snrcrt and snr[kk-1]<snrcrt and kk>=1 and kk<(lenspec-1):    # begin of continuously > crt
            w=w+1
        elif snr[kk]>=snrcrt and snr[kk-1]>=snrcrt and kk>1 and kk<(lenspec-1):   # continuously >= crt
            w=w+1
        elif snr[kk]<snrcrt and snr[kk-1]>=snrcrt and kk>1 and kk<=(lenspec-1):    # end of continuously > crt
            m.append(w)
            bindex.append(kk-w)
            eindex.append(kk-1)
            w=0
        elif snr[kk]<snrcrt and snr[kk-1]<snrcrt and kk>=1 and kk<=(lenspec-1):     # continuously < crt
            w=0
        elif snr[kk]>=snrcrt and snr[kk]>=snrcrt and kk==(lenspec-1):     # at last and continuously > crt
            w=w+1
            m.append(w)
            bindex.append(kk-w+1)
            eindex.append(kk)
        elif snr[kk]>=snrcrt and snr[kk]<snrcrt and kk==(lenspec-1):      # only last > crt
            w=1
            m.append(w)
            bindex.append(kk-w+1)
            eindex.append(kk)
    if len(m)==0:
        frange=0
        frmin=6
        frmax=6
        begind=0
        endind=0
        return begind,endind,frmin,frmax,frange
    ## FIND THE LONGEST SEGMENT
    longest=m.index(max(m))
    frmin=freq[bindex[longest]]
    frmax=freq[eindex[longest]]
    frange=frmax-frmin
##    print(frmin,frmax,frange,m)
    ## FAVOR THE SECOND LONGEST SEGMENT IF IT HAS LOWER FREQUENCY < 4 Hz AND LONGER
    ## THAN 1/4 OF THE LONGEST ONE
# =============================================================================
#     if len(m)>=2:
#         for mind in list(reversed(range(len(m)))):
#             mii=mind-len(m)
#             longest2=m.index(sorted(m)[mii])
#             frmin2=freq[bindex[longest2]]
#             if frmin2<=2.0:
#                 frmax2=freq[eindex[longest2]]
#                 frange2=frmax2-frmin2
# ##                print(frmin2,frmax2,frange2,snrcrtpara[1])
#                 if frmin2<frmin and 4*frange2>frange and frange2>snrcrtpara[1]:
#                     frmin=frmin2
#                     frmax=frmax2
#                     frange=frange2
#                     longest=longest2
#                     break
# =============================================================================
    begind=bindex[longest]
    endind=eindex[longest]
    ## EXTEND FREQUENCY BAND TO lowSNR
    if snrcrtpara[2]<snrcrtpara[0]:
# =============================================================================
#         if begind>ind1+1:
#             while snr[begind-1]<snr[begind] and snr[begind-1]>snrcrtpara[2] and begind-1>ind1+1:
#                 begind=begind-1
# =============================================================================
        if endind<ind2-1:
            while snr[endind+1]<snr[endind] and snr[endind+1]>snrcrtpara[2] and endind+1<ind2-1:
                endind=endind+1
        frmin=freq[begind]
        frmax=freq[endind]
        frange=frmax-frmin
    return begind,endind,frmin,frmax,frange

def fftplot(pwindata,dt,orid,sta):
## Do simple FFT on windowed waveform to compare the spectra with those after MTM
## Use SAC to replace this subrutine to do FFT might be better 
## USAGE: tstarsub.fftplot(PWINDATA,dt,orid,sta)
## INPUT:   pwindata  = P windowed data [0] and noise [1]
##          dt        = 1/sample rate
##          orid      = origin id
##          sta       = station name          
    for ii in range(pwindata.shape[0]):
        newspec=np.fft.fft(pwindata[ii])
        newfreq=np.fft.fftfreq(len(newspec),dt)
        newspec=np.sqrt(newspec)/(2*np.pi*newfreq)
        if ii==0:
            spec=newspec
            freq=newfreq
        else:
            n_spec=newspec
            n_freq=newfreq
    plt.figure(99)
    plt.clf()
    
# =============================================================================
#     plt.loglog(freq,spec,'b')
#     plt.loglog(n_freq,n_spec,'r')
#     plt.xlim([0,25])
#     plt.ylim([min(np.log(n_spec)),max(np.log(spec))+2000])
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Ap, nm/s')
#     plt.title('%s  Station: %s' % (orid,sta))
#     plt.savefig(g.figdir6+'/%s_%s_spectrum_loglog.pdf' % (orid,sta))
# =============================================================================
    
    plt.clf()
    plt.plot(freq,np.log(spec),'b')
    plt.plot(n_freq,np.log(n_spec),'r')
    plt.xlim([0,20])
    plt.ylim([-10,10])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ln(Ap), nm/s')
    plt.title('%s  Station: %s' % (orid,sta))
    plt.savefig(g.figdir6+'/%s_%s_spectrum.pdf' % (orid,sta))
    return    

def dospec(pwindata,swindata1,swindata2,dt,SDATA,orid,sta,snrcrtp,snrcrts,
           linresid,chan,doplot):
## Calculate amplitude spectrum of windowed waveform using multi-taper method
## USAGE: (spec_px,freq_px,spec_sx,freq_sx,spec,freq,n_spec,n_freq,frmn,frmx,
##         goodP1,goodS1)=tstarsub.dospec(PWINDATA,SWINDATA1,SWINDATA2,dt,
##                      SDATA,orid,sta,snrcrtp,snrcrts,lincor,chan,doplot)
## INPUT:   pwindata  = P windowed data [0] and noise [1]
##          swindata1 = S windowed data [0] and noise [1] and P coda [2] on channel 1
##          swindata2 = S windowed data [0] and noise [1] and P coda [2] on channel 2
##          dt        = 1/sample rate
##          SDATA     = existence of S arrival (Bool variable)
##          orid      = origin id
##          sta       = station name
##          snrcrtp   = minimum [SNR,width,lowSNR] of good P in freqency domain
##          snrcrts   = minimum [SNR,width,lowSNR] of good S in freqency domain
##          lincor    = MINIMUM LINEAR CORRELATION COEFFICIENTS
##          chan      = CHANNELS
##          doplot    = Bool variable for plotting spectrum
## OUTPUT:  spec_px = spectrum of good P signal
##          freq_px = freqency range of spectrum of good P signal
##          spec_sx = spectrum of good S signal
##          freq_sx = freqency range of spectrum of good S signal
##          spec    = spectrum of all signal
##          freq    = freqency range of spectrum of all signal
##          n_spec  = fspectrum of all noise
##          n_freq  = freqency range of spectrum of all noise
##          frmin   = minimum freqency of good P, [0]: P, [1]: S
##          frmax   = maximum freqency of good P, [0]: P, [1]: S
##          goodP   = Bool variable for good P
##          goodS   = Bool variable for good S
    ## PARAMETERS FOR MULTI-TAPER
    nft=1024
    npi=3.0
    smlen = 11
    residp=100
    resids=100
    pspec=[]
    s1spec=[]
    s2spec=[]
    pn_spec=[]
    s1n_spec=[]
    s2n_spec=[]
    pfreq=[]
    s1freq=[]
    s2freq=[]
    pn_freq=[]
    s1n_freq=[]
    s2n_freq=[]
    sch=-1
    ## DETERMINE P SPECTRA
    for ii in range(pwindata.shape[0]):
# =============================================================================
#         mtmresult=mtm.mtspec(pwindata[ii],dt,time_bandwidth=3.5,number_of_tapers=5)
# =============================================================================
        mtmresult=mtm.sine_psd(pwindata[ii],dt)
# =============================================================================
#         mtmresult=pymutt.mtft(pwindata[ii],dt=dt,npi=npi,
#                               nwin=int(npi*2-1),paddedlen=nft)
# =============================================================================
# =============================================================================
#         pspe=np.fft.fft(pwindata[ii])
#         pfre=np.fft.fftfreq(len(pspe),dt)
#         plt.clf()
#         plt.plot(pfre,np.log(pspe))
#         plt.show()
# =============================================================================
        newspec=mtmresult[0][1:]
        newfreq=mtmresult[1][1:]
# =============================================================================
#         newfreq=(mtmresult[1]*np.arange(len(mtmresult[0])))
# =============================================================================
        ## CONVERTING VELOCITY TO DISPLACEMENT BY DIVIDING BY 2*pi*f (Gubbins, p30)
        newspec=np.sqrt(newspec)/(2*np.pi*newfreq)
        
##        if smlen>0:
##            newspec=seis.smooth(newspec,smlen) ## SMOOTH THE SPECTRUM
        if ii==0:
            pspec=newspec
            pfreq=newfreq
            finterv=mtmresult[1]
        else:
##            newspec=seis.smooth(newspec,21) ## SMOOTH THE NOISE SPECTRUM
            pn_spec=newspec
            pn_freq=newfreq
# =============================================================================
#     print(snr)
#     print(snr[0])
#     print('spec[0] %f  n_spec[0] %f'%(spec[1], n_spec[1]))
# =============================================================================
    ## DETERMINE S SPECTRA ON CHANNEL 1
    for ii in range(swindata1.shape[0]):
# =============================================================================
#         mtmresult=mtm.mtspec(swindata1[ii],dt,time_bandwidth=3.5,number_of_tapers=5)
# =============================================================================
        mtmresult=mtm.sine_psd(swindata1[ii],dt)
# =============================================================================
#         mtmresult=pymutt.mtft(swindata1[ii],dt=dt,npi=npi,
#                                 nwin=int(npi*2-1),paddedlen=nft)
# =============================================================================
        newspec=mtmresult[0][1:]
#        print('s%d'%(len(newspec)))
        newfreq=mtmresult[1][1:]
        newspec=np.sqrt(newspec)/(2*np.pi*newfreq)
#        print('s%d'%(len(newspec)))
##        if smlen>0:
##            newspec=seis.smooth(newspec,smlen) ## SMOOTH THE SPECTRUM
        if ii==0:   ## S WAVE
            s1spec=newspec
            s1freq=newfreq
# =============================================================================
#             spec=np.vstack((spec,newspec))
#             freq=np.vstack((freq,newfreq))
#             finterv=[finterv,mtmresult[1]]
# =============================================================================
        elif ii==1: ## S NOISE
##            newspec=seis.smooth(newspec,21) ## SMOOTH THE NOISE SPECTRUM
            s1n_spec=newspec
            s1n_freq=newfreq
# =============================================================================
#             n_spec=np.vstack((n_spec,newspec))
#             n_freq=np.vstack((n_freq,newfreq))
# =============================================================================
        elif ii==2:  ## P CODA
            pcspec=newspec
    ## DETERMINE S SPECTRA ON CHANNEL 2
    for ii in range(swindata2.shape[0]):
# =============================================================================
#         mtmresult=mtm.mtspec(swindata2[ii],dt,time_bandwidth=3.5,number_of_tapers=5)
# =============================================================================
        mtmresult=mtm.sine_psd(swindata2[ii],dt)
# =============================================================================
#         mtmresult=pymutt.mtft(swindata1[ii],dt=dt,npi=npi,
#                                 nwin=int(npi*2-1),paddedlen=nft)
# =============================================================================
        newspec=mtmresult[0][1:]
        newfreq=mtmresult[1][1:]
        newspec=np.sqrt(newspec)/(2*np.pi*newfreq)
##        if smlen>0:
##            newspec=seis.smooth(newspec,smlen) ## SMOOTH THE SPECTRUM
        if ii==0:   ## S WAVE
            s2spec=newspec
            s2freq=newfreq
# =============================================================================
#             spec=np.vstack((spec,newspec))
#             freq=np.vstack((freq,newfreq))
#             finterv=[finterv,mtmresult[1]]
# =============================================================================
        elif ii==1: ## S NOISE
##            newspec=seis.smooth(newspec,21) ## SMOOTH THE NOISE SPECTRUM
            s2n_spec=newspec
            s2n_freq=newfreq
# =============================================================================
#             n_spec=np.vstack((n_spec,newspec))
#             n_freq=np.vstack((n_freq,newfreq))
# =============================================================================
## USE P CODA
##        if ii==0:   ## S WAVE
##            sspec=newspec
##        elif ii==1: ## S NOISE
####            newspec=seis.smooth(newspec,21) ## SMOOTH THE NOISE SPECTRUM
##            n_spec=np.vstack((n_spec,newspec))
##            n_freq=np.vstack((n_freq,newfreq))
##        elif ii==2:  ## P CODA
####            pcspec=seis.smooth(newspec,21)  ## SMOOTH THE P CODA SPECTRUM
##            pcspec=newspec
##            spcspec=np.sqrt((sspec**2-pcspec**2).clip(min=0))
##            spec=np.vstack((spec,spcspec))
##            freq=np.vstack((freq,newfreq))
##            finterv=[finterv,mtmresult['df']]
    spec_px=pspec
    freq_px=pfreq
    spec_sx=s1spec
    freq_sx=s1freq
    frmin=[6,6]
    frmax=[6,6]
    goodP=False
    goodS=False
    nsamp=[0,0]
    snrmed=[0,0]
    ## SINGAL-TO-NOISE RATIO
# =============================================================================
#     snr=spec/n_spec
# =============================================================================
# =============================================================================
#     if sta == 'SP11':
#         print(pn_spec)
# =============================================================================
    psnr=pspec/pn_spec
    s1snr=s1spec/s1n_spec
    s2snr=s2spec/s2n_spec
    if smlen>0:
        psnr=seis.smooth(pspec,smlen)/seis.smooth(pn_spec,smlen)
# =============================================================================
#         for ii in [0]:
#             snr[ii]=seis.smooth(spec[ii],smlen)/seis.smooth(n_spec[ii],smlen)
# =============================================================================
##    lenspec=snr.shape[1]
    (begind,endind,frminp,frmaxp,frangep)=longseg(psnr,snrcrtp,pfreq)
    frmin[0]=frminp
    frmax[0]=frmaxp
    if frangep<snrcrtp[1] or frminp>4:
##        print('notgoodP2snr')
        goodP=False
        goodS=False
        return sch,residp,resids,spec_px,freq_px,spec_sx,freq_sx,pspec,s1spec,s2spec,pfreq,s1freq,s2freq,pn_spec,s1n_spec,s2n_spec,pn_freq,s1n_freq,s2n_freq,frmin,frmax,goodP,goodS
    else:
        goodP=True
    spec_px=pspec[begind:endind]
    freq_px=pfreq[begind:endind]
    spec_nx=pn_spec[begind:endind]
##    spec_px=np.sqrt(spec_px**2-spec_nx**2)
    nsamp[0]=freq_px.shape[0]
    snr_px=psnr[begind:endind]
    snrmed[0]=float(np.median(snr_px))
    coeffp=np.polyfit(freq_px,np.log(spec_px),1)
# =============================================================================
#     print(coeffp)
# =============================================================================
    synp=coeffp[1]+freq_px*coeffp[0]
##    residp=np.linalg.norm(np.log(synp)-np.log(spec_px))/np.sqrt(len(freq_px)-1)
    residp=seis.lincorrcoef(freq_px,np.log(spec_px))
    
    if coeffp[0]<0 and abs(residp)>=linresid[0]:
        goodP=True
    else:
##        print('notgoodP2lin coeffp %f residp %f'%(coeffp[0], residp))
        goodP=False
        goodS=False
        return sch,residp,resids,spec_px,freq_px,spec_sx,freq_sx,pspec,s1spec,s2spec,pfreq,s1freq,s2freq,pn_spec,s1n_spec,s2n_spec,pn_freq,s1n_freq,s2n_freq,frmin,frmax,goodP,goodS
    ## FIND THE LONGEST SEGMENT OF S SPECTRUM WITH SNR > SNRCRT
    if SDATA:
##        print('SDATA')
        (begind1,endind1,frmins1,frmaxs1,franges1)=longseg(s1snr,snrcrts,s1freq)
        (begind2,endind2,frmins2,frmaxs2,franges2)=longseg(s2snr,snrcrts,s2freq)
        if franges1>=franges2:
            begind =begind1
            endind =endind1
            frmins =frmins1
            frmaxs =frmaxs1
            franges=franges1
            sch=1
        else:
            begind =begind2
            endind =endind2
            frmins =frmins2
            frmaxs =frmaxs2
            franges=franges2
            sch=2
        frmin[1]=frmins
        frmax[1]=frmaxs
##        print(sta,frmins,frmaxs)
        if franges<snrcrts[1] or frmins>4:
            goodS=False
##            return spec_px,freq_px,spec_sx,freq_sx,spec,freq,n_spec,n_freq,frmin,frmax,goodP,goodS
        elif sch ==1 :
            goodS=True
            spec_sx=s1spec[begind:endind]
            freq_sx=s1freq[begind:endind]
            spec_nx=s1n_spec[begind:endind]
    ##        spec_sx=np.sqrt(spec_sx**2-spec_nx**2)
            nsamp[1]=freq_sx.shape[0]
            snr_sx=s1snr[begind:endind]
            snrmed[1]=float(np.median(snr_sx))
            coeffs=np.polyfit(freq_sx,np.log(spec_sx),1)
            syns=coeffs[1]+freq_sx*coeffs[0]
    ##        resids=np.linalg.norm(np.log(syns)-np.log(spec_sx))/np.sqrt(len(freq_sx)-1)
            resids=seis.lincorrcoef(freq_sx,np.log(spec_sx))
            if coeffs[0]<0 and abs(resids)>=linresid[1]:
                goodS=True
            else:
                goodS=False
        elif sch ==2 :
            goodS=True
            spec_sx=s2spec[begind:endind]
            freq_sx=s2freq[begind:endind]
            spec_nx=s2n_spec[begind:endind]
    ##        spec_sx=np.sqrt(spec_sx**2-spec_nx**2)
            nsamp[1]=freq_sx.shape[0]
            snr_sx=s2snr[begind:endind]
            snrmed[1]=float(np.median(snr_sx))
            coeffs=np.polyfit(freq_sx,np.log(spec_sx),1)
            syns=coeffs[1]+freq_sx*coeffs[0]
    ##        resids=np.linalg.norm(np.log(syns)-np.log(spec_sx))/np.sqrt(len(freq_sx)-1)
            resids=seis.lincorrcoef(freq_sx,np.log(spec_sx))
            if coeffs[0]<0 and abs(resids)>=linresid[1]:
                goodS=True
            else:
                goodS=False 
                
    if doplot:
        if SDATA and goodS:
            ## PLOT P AND S SIGNAL AND FITTING
            plt.figure(2)
            plt.clf()
            plt.subplot(2,2,1)
            plt.plot(pfreq,np.log(pspec),'b')   # P WAVE
            plt.plot(pn_freq,np.log(pn_spec),'r')
            plt.plot(freq_px,np.log(spec_px),'k')
            plt.plot([frmin[0],frmin[0]],np.log([min(pn_spec),max(pspec)]),'g')
            plt.plot([frmax[0],frmax[0]],np.log([min(pn_spec),max(pspec)]),'g')
            plt.plot(freq_px,synp,'g--',linewidth=2)
##            plt.loglog(freq[0],spec[0],'b')   # P WAVE
##            plt.loglog(n_freq[0],n_spec[0],'r')
##            plt.loglog(freq_px,spec_px,'k')
##            plt.loglog([frmin[0],frmin[0]],[min(n_spec[0]),max(spec[0])],'g')
##            plt.loglog([frmax[0],frmax[0]],[min(n_spec[0]),max(spec[0])],'g')
##            plt.loglog(freq_px,synp,'g--',linewidth=2)
            plt.text(10,max(np.log(pspec))/2,'slope = %.4f' % coeffp[0])
            plt.text(10,max(np.log(pspec))/5,'lincorr = %.4f' % residp)
            plt.xlim([0.05,20])
            plt.ylabel('ln(Ap) on '+chan[2]+', nm/s')
            plt.title('Station: %s' % sta)            
            plt.subplot(2,2,2)
            plt.plot(pfreq,psnr)
            plt.axhline(snrcrtp[0],color='g',linestyle='--')
            plt.axvline(frmin[0],color='g')
            plt.axvline(frmax[0],color='g')
            plt.xlim([0,20])
            plt.ylim([0,250])
            plt.ylabel('P Signal-to-Noise Ratio')
            plt.title('%s' % orid)            
            plt.subplot(2,2,3)
            if sch == 1:
                plt.plot(s1freq,np.log(s1spec),'b')   # S WAVE
            
##            plt.plot(freq[0],np.log(sspec),'y')   # S WAVE
##            plt.plot(freq[0],np.log(pcspec),'y--')   # P CODA
                plt.plot(s1n_freq,np.log(s1n_spec),'r') # NOISE
            elif sch == 2:
                plt.plot(s2freq,np.log(s2spec),'b')   # S WAVE
                plt.plot(s2n_freq,np.log(s2n_spec),'r') # NOISE
########            plt.plot(freq2[1],np.log(spec2[1]),'g')
########            plt.plot(n_freq2[1],np.log(n_spec2[1]),'g--')
            plt.plot(freq_sx,np.log(spec_sx),'k')
            plt.plot([frmin[1],frmin[1]],np.log([min(s1n_spec),max(s1spec)]),'g')
            plt.plot([frmax[1],frmax[1]],np.log([min(s1n_spec),max(s1spec)]),'g')
            plt.plot(freq_sx,syns,'g--',linewidth=2)
##            plt.loglog(freq[1],spec[1],'b')   # S - P CODA
####            plt.loglog(freq[0],sspec,'y')   # S WAVE
##            plt.loglog(freq[0],pcspec,'y--')   # P CODA
##            plt.loglog(n_freq[1],n_spec[1],'r')
##            plt.loglog(freq_sx,spec_sx,'k')
##            plt.loglog([frmin[1],frmin[1]],[min(n_spec[1]),max(spec[1])],'g')
##            plt.loglog([frmax[1],frmax[1]],[min(n_spec[1]),max(spec[1])],'g')
##            plt.loglog(freq_sx,syns,'g--',linewidth=2)
            plt.xlim([0.05,20])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('ln(As) on '+chan[sch-1]+', nm/s')
            plt.text(10,max(np.log(s1spec))/2,'slope = %.4f' % coeffs[0])
            plt.text(10,max(np.log(s1spec))/5,'lincorr = %.4f' % resids)
            plt.subplot(2,2,4)
            if sch == 1:
                plt.plot(s1freq,s1snr)
            elif sch == 2:
                plt.plot(s2freq,s2snr)
            plt.axhline(snrcrts[0],color='g',linestyle='--')
            plt.axvline(frmin[1],color='g')
            plt.axvline(frmax[1],color='g')
            plt.ylim([0,30])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('S Signal-to-Noise Ratio')
            plt.savefig(g.figdir2+'/%s_%s_PSsnr.eps' % (orid,sta))
        else:
            ## PLOT ONLY P SIGNAL AND FITTING
            plt.figure(1)
            plt.clf()
            plt.subplot(2,2,1)
            plt.plot(pfreq,np.log(pspec),'b')
########        plt.plot(freq2[0],np.log(spec2[0]),'g')
########        plt.plot(n_freq2[0],np.log(n_spec2[0]),'g--')
            plt.plot(pn_freq,np.log(pn_spec),'r')
            plt.plot(freq_px,np.log(spec_px),'k')
            plt.plot([frmin[0],frmin[0]],np.log([min(pn_spec),max(pspec)]),'g')
            plt.plot([frmax[0],frmax[0]],np.log([min(pn_spec),max(pspec)]),'g')
            plt.plot(freq_px,synp,'g--',linewidth=2)
            plt.text(10,max(np.log(pspec))/2,'slope = %.4f' % coeffp[0])
            plt.text(10,max(np.log(pspec))/10,'lincorr = %.4f' % residp)
########        plt.plot(matfreq_px,np.log(matspec_px),'k')
            plt.xlim([0,20])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('ln(Ap) on '+chan[2]+', nm/s')
            plt.title('Station: %s' % sta)
            plt.subplot(2,2,2)
            plt.plot(pfreq,psnr)
            plt.axhline(snrcrtp[0],color='g',linestyle='--')
            plt.axvline(frmin[0],color='g')
            plt.axvline(frmax[0],color='g')
            plt.xlim([0,20])
            plt.ylim([0,250])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('P Signal-to-Noise Ratio')
            plt.title('%s' % orid)
            plt.subplot(2,2,3)
            if sch == 1:
                plt.plot(s1freq,np.log(s1spec),'b')   # S WAVE
                plt.plot(s1n_freq,np.log(s1n_spec),'r') # NOISE
            elif sch == 2:
                plt.plot(s2freq,np.log(s2spec),'b')   # S WAVE
                plt.plot(s2n_freq,np.log(s2n_spec),'r') # NOISE
            plt.xlim([0.05,20])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('ln(As) on '+chan[sch-1]+', nm/s')
            plt.subplot(2,2,4)
            if sch == 1:
                plt.plot(s1freq,s1snr)
            elif sch == 2:
                plt.plot(s2freq,s2snr)
            plt.axhline(snrcrts[0],color='g',linestyle='--')
            plt.axvline(frmin[1],color='g')
            plt.axvline(frmax[1],color='g')
            plt.ylim([0,30])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('S Signal-to-Noise Ratio')
            plt.savefig(g.figdir2+'/%s_%s_Psnr.eps' % (orid,sta))

    return sch,residp,resids,spec_px,freq_px,spec_sx,freq_sx,pspec,s1spec,s2spec,pfreq,s1freq,s2freq,pn_spec,s1n_spec,s2n_spec,pn_freq,s1n_freq,s2n_freq,frmin,frmax,goodP,goodS


def plotspec(plotspecloglog,fitting,fitting1,fitting2,fitting3,lincorr,saving,sta,orid,POS,lnmomen,fc,alpha,icase,sitedata=0):
## PLOT AMPLITUDE SPECTRUM
    if POS.upper()=='P':
        ind=0
        sch=0
        xmax=20
        ymin=-10
        ymax=10
        textx=6
    elif POS.upper()=='S':
        ind=1
        sch=saving['sch']
        xmax=20
        ymin=-10
        ymax=10
        textx=2.5
    else:
        raise ValueError("P or S wave?")
    chan=['Z','E','N']
    corr=saving['corr'][ind]
    spec=saving['spec'][sch]
    freq=saving['freq'][sch]
    n_spec=saving['nspec'][ind]
    n_freq=saving['nfreq'][ind]
    frmin=saving[2]['frmin'][ind]
    frmax=saving[2]['frmax'][ind]
    invtstar=saving[icase]['tstar'][ind]
    synspec=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar)/(1+(freq/fc)**2))
# =============================================================================
#     if sta == 'MALA':
#         print(len(spec))
#         print(ind)
#         print('synnnnnnnn')
#         print(synspec)
#         print('specccccc')
#         print(spec)
# =============================================================================
    if POS.upper()=='S':
        invtstarP=saving[icase]['tstar'][0]
        ttP=saving['Ptt']
        ttS=saving['Stt']
        QpQs=2.25
        invtstar2=invtstarP*QpQs*ttS/ttP
        synspec2=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar2)/(1+(freq/fc)**2))
        QpQs=1.75
        invtstar2=invtstarP*QpQs*ttS/ttP
        synspec3=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar2)/(1+(freq/fc)**2))
    indx=np.all([(freq>=frmin),(freq<frmax)],axis=0)
    specx=spec[indx]
    freqx=freq[indx]
    synx=synspec[indx]
    resid=(1-((np.linalg.norm(np.log(synx)-np.log(specx)))**2/(len(freqx)-1)
             /np.var(np.log(specx))))    
    df=abs(freq[1]-freq[0])
    nlowf=0
    narea=0
    for ifreq in range(len(freq)):
        if (freq[ifreq]>frmax and freq[ifreq]<15):
            if (np.log(synspec[ifreq])<np.log(spec[ifreq]) or
                 np.log(synspec[ifreq])>np.log(spec[ifreq])+1):
                narea=narea+np.log(spec[ifreq])-np.log(synspec[ifreq])
            if np.log(synspec[ifreq])>np.log(spec[ifreq])+2:
                nlowf=nlowf+5
            elif np.log(synspec[ifreq])>np.log(spec[ifreq])+1:
                nlowf=nlowf+1
    if narea<-10 and nlowf*df>3:
        resid=0
##    if POS.upper()=='P':
##        est=saving[icase]['est'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)
##        dat=saving[icase]['dat'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)
##    elif POS.upper()=='S':
##        est=saving[icase]['est'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)+lnmomen
##        dat=saving[icase]['dat'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)+lnmomen
##    (fmspec,fmfreq)=fm.fm(orid,sta)
    plt.figure(99)
    plt.clf()
    
#####################remain to be studied########################
    if not isinstance(sitedata, int):
        plt.plot(freq,np.log(spec)-sitedata,'b--')   
#################################################################
    if plotspecloglog:
        plt.clf()
        plt.loglog(freq,spec,'b')
        plt.loglog(n_freq,n_spec,'r')
        plt.loglog([frmin,frmin],[min(n_spec),max(spec)],'g')
        plt.loglog([frmax,frmax],[min(n_spec),max(spec)],'g')
        plt.loglog(freq,synspec,'g--',linewidth=2)
##        plt.loglog(fmfreq,fmspec,'y--',linewidth=2)
# =============================================================================
#         if POS.upper()=='S':
#             plt.loglog(freq,synspec2,'k:',linewidth=2)
#             plt.loglog(freq,synspec3,'k--',linewidth=2)
# =============================================================================
        plt.text(3,1,'t* = %.2f' % invtstar)
        plt.text(0.5,pow(10,0),'fitting1 = %.4f' % fitting)
        plt.text(0.5,pow(10,-1),'fitting2 = %.4f' % fitting1)
        plt.text(0.5,pow(10,-2),'fitting3 = %.4f' % fitting2)
        plt.text(0.5,pow(10,-3),'fitting4 = %.4f' % fitting3)
# =============================================================================
#         plt.text(0.5,pow(10,-2),'misfit = %.4f' % saving[icase]['misfit'][ind])
#         plt.text(0.5,pow(10,-3),'lincorr = %.4f' % lincorr)
# =============================================================================
        plt.text(0.5,pow(10,-4),'frange =[%.4f, %.4f]' % (frmin, frmax))
        plt.text(0.5,pow(10,-5),'sd = %.4f' % saving[icase]['err'][ind])
        plt.xlim([0,25])
#        plt.ylim([min(np.log(n_spec)),max(np.log(spec))+10000000])
##        plt.ylim(np.log([min(n_spec),max(spec)]))

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('A%s on BH%s, nm/s' %(POS.lower(),chan[sch]))
        plt.title('%s  Station: %s' % (orid,sta))
        plt.savefig(g.figdir3+'/%s_%s_%sspectrum_loglog.pdf' % (orid,sta,POS.upper()))
        

        return
    else:
        plt.plot(freq,np.log(spec),'b')
        plt.plot(n_freq,np.log(n_spec),'r')
        plt.plot([frmin,frmin],np.log([min(n_spec),max(spec)]),'g')
        plt.plot([frmax,frmax],np.log([min(n_spec),max(spec)]),'g')
        plt.plot(freq,np.log(synspec),'g--',linewidth=2)
##        plt.plot(fmfreq,np.log(fmspec),'y--',linewidth=2)
# =============================================================================
#         if POS.upper()=='S':
#             plt.plot(freq,np.log(synspec2),'k:',linewidth=2)
#             plt.plot(freq,np.log(synspec3),'k--',linewidth=2)
# =============================================================================
        plt.text(textx,4,'t* = %.2f' % invtstar)
        plt.text(10,9,'fitting1 = %.4f' % fitting)
        plt.text(10,7.5,'fitting2 = %.4f' % fitting1)
        plt.text(10,6,'fitting3 = %.4f' % fitting2)
        plt.text(10,4.5,'fitting4 = %.4f' % fitting3)
# =============================================================================
#         plt.text(10,7.5,'curve fitting = %.4f' % resid)
# =============================================================================
# =============================================================================
#         plt.text(10,6,'misfit = %.4f' % saving[icase]['misfit'][ind])
#         plt.text(10,4.5,'lincorr = %.4f' % lincorr)
# =============================================================================
        plt.text(10,3,'frange =[%.4f, %.4f]' % (frmin, frmax))
        plt.text(10,1.5,'sd = %.4f' % saving[icase]['err'][ind])
        plt.xlim([0,xmax])
#        plt.ylim([ymin,ymax+3])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ln(A%s) on BH%s, nm/s' %(POS.lower(),chan[sch]))
        plt.title('%s  Station: %s' % (orid,sta))
        plt.savefig(g.figdir3+'/%s_%s_%sspectrum.pdf' % (orid,sta,POS.upper()))
        
        return
##    plt.plot(freqx,est,'k--')
##    plt.plot(freqx,dat,'k')

#   plt.ylim(np.log([min(n_spec),max(spec)]))


#    plt.ylim([-5,np.log(max(spec))])
# =============================================================================
#     plt.ylim([10**(int(np.log(min(min(spec),min(n_spec))))-2),
#                    10**(int(np.log(max(max(spec),max(n_spec))))+1)])
# =============================================================================

def plotspec1(plotspecloglog,fitting,fitting1,fitting2,fitting3,lincorr,saving,sta,orid,POS,lnmomen,fc,alpha,icase,sitedata=0):
## PLOT AMPLITUDE SPECTRUM
    if POS.upper()=='P':
        sitefreq = np.linspace(0.333,14.333,43)
        for i in range(len(sitefreq)):
            sitefreq[i]=format(sitefreq[i],'.4f')
        ind=0
        sch=0
        xmax=20
        ymin=-10
        ymax=10
        textx=6
    elif POS.upper()=='S':
        sitefreq = np.linspace(0.200,14.600,73)
        for i in range(len(sitefreq)):
            sitefreq[i]=format(sitefreq[i],'.4f')
        ind=1
        sch=saving['sch']
        xmax=20
        ymin=-10
        ymax=10
        textx=7
    else:
        raise ValueError("P or S wave?")
    chan=['Z','E','N']
    corr=saving['corr'][ind]
    spec=saving['spec'][sch]
    freq=saving['freq'][sch]
    n_spec=saving['nspec'][ind]
    n_freq=saving['nfreq'][ind]
    frmin=saving[2]['frmin'][ind]
    frmax=saving[2]['frmax'][ind]
    invtstar=saving[icase]['tstar'][ind]
    atten = float(saving[icase]['aveATTEN'][ind])/1000
    synspec=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar)/(1+(freq/fc)**2))
# =============================================================================
#     if sta == 'MALA':
#         print(len(spec))
#         print(ind)
#         print('synnnnnnnn')
#         print(synspec)
#         print('specccccc')
#         print(spec)
# =============================================================================
    if POS.upper()=='S':
        invtstarP=saving[icase]['tstar'][0]
        ttP=saving['Ptt']
        ttS=saving['Stt']
        QpQs=2.25
        invtstar2=invtstarP*QpQs*ttS/ttP
        synspec2=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar2)/(1+(freq/fc)**2))
        QpQs=1.75
        invtstar2=invtstarP*QpQs*ttS/ttP
        synspec3=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar2)/(1+(freq/fc)**2))
    indx=np.all([(freq>=frmin),(freq<frmax)],axis=0)
    specx=spec[indx]
    freqx=freq[indx]
    synx=synspec[indx]
    resid=(1-((np.linalg.norm(np.log(synx)-np.log(specx)))**2/(len(freqx)-1)
             /np.var(np.log(specx))))    
    df=abs(freq[1]-freq[0])
    nlowf=0
    narea=0
    for ifreq in range(len(freq)):
        if (freq[ifreq]>frmax and freq[ifreq]<15):
            if (np.log(synspec[ifreq])<np.log(spec[ifreq]) or
                 np.log(synspec[ifreq])>np.log(spec[ifreq])+1):
                narea=narea+np.log(spec[ifreq])-np.log(synspec[ifreq])
            if np.log(synspec[ifreq])>np.log(spec[ifreq])+2:
                nlowf=nlowf+5
            elif np.log(synspec[ifreq])>np.log(spec[ifreq])+1:
                nlowf=nlowf+1
    if narea<-10 and nlowf*df>3:
        resid=0
##    if POS.upper()=='P':
##        est=saving[icase]['est'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)
##        dat=saving[icase]['dat'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)
##    elif POS.upper()=='S':
##        est=saving[icase]['est'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)+lnmomen
##        dat=saving[icase]['dat'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)+lnmomen
##    (fmspec,fmfreq)=fm.fm(orid,sta)
    plt.figure(99)
    plt.clf()
    

    spec_x = []
    ind1=np.nonzero(abs(freq-sitefreq[0])<0.01)[0][0]
    ind2=ind1+len(sitefreq)
    spec_x = spec[ind1:ind2]
    
#####################remain to be studied########################

# =============================================================================
#     if not isinstance(sitedata, int):
#         plt.plot(freq,np.log(spec)-sitedata,'b--')   
# =============================================================================
#################################################################
    if plotspecloglog:
        plt.clf()
        plt.loglog(freq,spec,'b')
        plt.loglog(n_freq,n_spec,'r')
        plt.loglog([frmin,frmin],[min(n_spec),max(spec)],'g')
        plt.loglog([frmax,frmax],[min(n_spec),max(spec)],'g')
        plt.loglog(freq,synspec,'g--',linewidth=2)
        plt.loglog(sitefreq,spec_x/(np.exp(sitedata)),color='skyblue',linestyle='--',linewidth=2)
##        plt.loglog(fmfreq,fmspec,'y--',linewidth=2)
# =============================================================================
#         if POS.upper()=='S':
#             plt.loglog(freq,synspec2,'k:',linewidth=2)
#             plt.loglog(freq,synspec3,'k--',linewidth=2)
# =============================================================================
        plt.text(3,1,'t* = %.2f' % invtstar)
        plt.text(0.5,pow(10,0),'fitting1 = %.4f' % fitting)
        plt.text(0.5,pow(10,-1),'fitting2 = %.4f' % fitting1)
        plt.text(0.5,pow(10,-2),'fitting3 = %.4f' % fitting2)
        plt.text(0.5,pow(10,-3),'fitting4 = %.4f' % fitting3)
# =============================================================================
#         plt.text(0.5,pow(10,-2),'misfit = %.4f' % saving[icase]['misfit'][ind])
#         plt.text(0.5,pow(10,-3),'lincorr = %.4f' % lincorr)
# =============================================================================
        plt.text(0.5,pow(10,-4),'frange =[%.4f, %.4f]' % (frmin, frmax))
        plt.text(0.5,pow(10,-5),'sd = %.4f' % saving[icase]['err'][ind])
        plt.xlim([0,25])
#        plt.ylim([min(np.log(n_spec)),max(np.log(spec))+10000000])
##        plt.ylim(np.log([min(n_spec),max(spec)]))

# =============================================================================
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('A%s on BH%s, nm/s' %(POS.lower(),chan[sch]))
#         plt.title('%s  Station: %s' % (orid,sta))
# =============================================================================
        plt.savefig(g.figdir3+'/%s_%s_%sspectrum_loglog.pdf' % (orid,sta,POS.upper()))

        return
    else:
        plt.plot(freq,np.log(spec),'b')
        plt.plot(n_freq,np.log(n_spec),'r')
        plt.plot([frmin,frmin],np.log([min(n_spec),max(spec)]),'g')
        plt.plot([frmax,frmax],np.log([min(n_spec),max(spec)]),'g')
        plt.plot(sitefreq,np.log(spec_x)-sitedata,color='skyblue',linestyle='--',linewidth=2)
        plt.plot(freq,np.log(synspec),'g--',linewidth=2)
##        plt.plot(fmfreq,np.log(fmspec),'y--',linewidth=2)
# =============================================================================
#         if POS.upper()=='S':
#             plt.plot(freq,np.log(synspec2),'k:',linewidth=2)
#             plt.plot(freq,np.log(synspec3),'k--',linewidth=2)
# =============================================================================
        plt.text(11,textx,'t* = %.2f' % invtstar)
        plt.text(11,textx-1,'1/Q = %.5f' % atten)
# =============================================================================
#         plt.text(10,9,'fitting1 = %.4f' % fitting)
#         plt.text(10,7.5,'fitting2 = %.4f' % fitting1)
#         plt.text(10,6,'fitting3 = %.4f' % fitting2)
#         plt.text(10,4.5,'fitting4 = %.4f' % fitting3)
# =============================================================================
# =============================================================================
#         plt.text(10,7.5,'curve fitting = %.4f' % resid)
# =============================================================================
# =============================================================================
#         plt.text(10,6,'misfit = %.4f' % saving[icase]['misfit'][ind])
#         plt.text(10,4.5,'lincorr = %.4f' % lincorr)
# =============================================================================
# =============================================================================
#         plt.text(10,3,'frange =[%.4f, %.4f]' % (frmin, frmax))
#         plt.text(10,1.5,'sd = %.4f' % saving[icase]['err'][ind])
# =============================================================================
        plt.xlim([0,xmax])
#        plt.ylim([ymin,ymax+3])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ln(A%s) on BH%s, nm/s' %(POS.lower(),chan[sch]))
        plt.title('%s  Station: %s' % (orid,sta))
        plt.savefig(g.figdir3+'/%s_%s_%sspectrum.pdf' % (orid,sta,POS.upper()))
        
        return
##    plt.plot(freqx,est,'k--')
##    plt.plot(freqx,dat,'k')

#   plt.ylim(np.log([min(n_spec),max(spec)]))


#    plt.ylim([-5,np.log(max(spec))])
# =============================================================================
#     plt.ylim([10**(int(np.log(min(min(spec),min(n_spec))))-2),
#                    10**(int(np.log(max(max(spec),max(n_spec))))+1)])
# =============================================================================






def calresspec(saving,sta,orid,POS,lnmomen,fc,alpha):
## CALCULATE RESIDUAL SPECTRUM FOR EACH STATION
    if POS.upper()=='P':
        ind=0
    elif POS.upper()=='S':
        ind=1
    else:
        raise ValueError("P or S wave?")
    freq_x = saving[2][POS.lower()][0]
    
    spec_x = saving[2][POS.lower()][1]
    correc = saving['corr'][ind]
    invtstar=saving[3]['tstar'][ind]
    righthand = lnmomen-np.pi*freq_x*(freq_x**(-alpha)*invtstar)
    resspec = np.array([np.log(spec_x)-np.log(correc)
                  +np.log(1+(freq_x/fc)**2)-righthand])
    resratio = resspec/righthand*100
    resspec = np.vstack((freq_x,resspec))
    resspec = np.vstack((resspec,resratio))
    resspec = resspec.transpose()
#     exit()
    return resspec


def plotspec1(plotspecloglog,fitting,fitting1,fitting2,fitting3,lincorr,saving,sta,orid,POS,lnmomen,fc,alpha,icase,sitedata=0):
## PLOT AMPLITUDE SPECTRUM
    if POS.upper()=='P':
        sitefreq = np.linspace(0.333,14.333,43)
        for i in range(len(sitefreq)):
            sitefreq[i]=format(sitefreq[i],'.4f')
        ind=0
        sch=0
        xmax=20
        ymin=-10
        ymax=10
        textx=6
    elif POS.upper()=='S':
        sitefreq = np.linspace(0.200,14.600,73)
        for i in range(len(sitefreq)):
            sitefreq[i]=format(sitefreq[i],'.4f')
        ind=1
        sch=saving['sch']
        xmax=20
        ymin=-10
        ymax=10
        textx=7
    else:
        raise ValueError("P or S wave?")
    chan=['Z','E','N']
    corr=saving['corr'][ind]
    spec=saving['spec'][sch]
    freq=saving['freq'][sch]
    n_spec=saving['nspec'][ind]
    n_freq=saving['nfreq'][ind]
    frmin=saving[2]['frmin'][ind]
    frmax=saving[2]['frmax'][ind]
    invtstar=saving[icase]['tstar'][ind]
    atten = float(saving[icase]['aveATTEN'][ind])/1000
    synspec=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar)/(1+(freq/fc)**2))
# =============================================================================
#     if sta == 'MALA':
#         print(len(spec))
#         print(ind)
#         print('synnnnnnnn')
#         print(synspec)
#         print('specccccc')
#         print(spec)
# =============================================================================
    if POS.upper()=='S':
        invtstarP=saving[icase]['tstar'][0]
        ttP=saving['Ptt']
        ttS=saving['Stt']
        QpQs=2.25
        invtstar2=invtstarP*QpQs*ttS/ttP
        synspec2=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar2)/(1+(freq/fc)**2))
        QpQs=1.75
        invtstar2=invtstarP*QpQs*ttS/ttP
        synspec3=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar2)/(1+(freq/fc)**2))
    indx=np.all([(freq>=frmin),(freq<frmax)],axis=0)
    specx=spec[indx]
    freqx=freq[indx]
    synx=synspec[indx]
    resid=(1-((np.linalg.norm(np.log(synx)-np.log(specx)))**2/(len(freqx)-1)
             /np.var(np.log(specx))))    
    df=abs(freq[1]-freq[0])
    nlowf=0
    narea=0
    for ifreq in range(len(freq)):
        if (freq[ifreq]>frmax and freq[ifreq]<15):
            if (np.log(synspec[ifreq])<np.log(spec[ifreq]) or
                 np.log(synspec[ifreq])>np.log(spec[ifreq])+1):
                narea=narea+np.log(spec[ifreq])-np.log(synspec[ifreq])
            if np.log(synspec[ifreq])>np.log(spec[ifreq])+2:
                nlowf=nlowf+5
            elif np.log(synspec[ifreq])>np.log(spec[ifreq])+1:
                nlowf=nlowf+1
    if narea<-10 and nlowf*df>3:
        resid=0
##    if POS.upper()=='P':
##        est=saving[icase]['est'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)
##        dat=saving[icase]['dat'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)
##    elif POS.upper()=='S':
##        est=saving[icase]['est'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)+lnmomen
##        dat=saving[icase]['dat'][ind].transpose()[0]-np.log(1+(freqx/fc)**2)+np.log(corr)+lnmomen
##    (fmspec,fmfreq)=fm.fm(orid,sta)
    plt.figure(99)
    plt.clf()
    

    spec_x = []
    ind1=np.nonzero(abs(freq-sitefreq[0])<0.01)[0][0]
    ind2=ind1+len(sitefreq)
    spec_x = spec[ind1:ind2]
    
#####################remain to be studied########################

# =============================================================================
#     if not isinstance(sitedata, int):
#         plt.plot(freq,np.log(spec)-sitedata,'b--')   
# =============================================================================
#################################################################
    if plotspecloglog:
        plt.clf()
        plt.loglog(freq,spec,'b')
        plt.loglog(n_freq,n_spec,'r')
        plt.loglog([frmin,frmin],[min(n_spec),max(spec)],'g')
        plt.loglog([frmax,frmax],[min(n_spec),max(spec)],'g')
        plt.loglog(freq,synspec,'g--',linewidth=2)
        plt.loglog(sitefreq,spec_x/(np.exp(sitedata)),color='skyblue',linestyle='--',linewidth=2)
##        plt.loglog(fmfreq,fmspec,'y--',linewidth=2)
# =============================================================================
#         if POS.upper()=='S':
#             plt.loglog(freq,synspec2,'k:',linewidth=2)
#             plt.loglog(freq,synspec3,'k--',linewidth=2)
# =============================================================================
        plt.text(3,1,'t* = %.2f' % invtstar)
        plt.text(0.5,pow(10,0),'fitting1 = %.4f' % fitting)
        plt.text(0.5,pow(10,-1),'fitting2 = %.4f' % fitting1)
        plt.text(0.5,pow(10,-2),'fitting3 = %.4f' % fitting2)
        plt.text(0.5,pow(10,-3),'fitting4 = %.4f' % fitting3)
# =============================================================================
#         plt.text(0.5,pow(10,-2),'misfit = %.4f' % saving[icase]['misfit'][ind])
#         plt.text(0.5,pow(10,-3),'lincorr = %.4f' % lincorr)
# =============================================================================
        plt.text(0.5,pow(10,-4),'frange =[%.4f, %.4f]' % (frmin, frmax))
        plt.text(0.5,pow(10,-5),'sd = %.4f' % saving[icase]['err'][ind])
        plt.xlim([0,25])
#        plt.ylim([min(np.log(n_spec)),max(np.log(spec))+10000000])
##        plt.ylim(np.log([min(n_spec),max(spec)]))

# =============================================================================
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('A%s on BH%s, nm/s' %(POS.lower(),chan[sch]))
#         plt.title('%s  Station: %s' % (orid,sta))
# =============================================================================
        plt.savefig(g.figdir3+'/%s_%s_%sspectrum_loglog.pdf' % (orid,sta,POS.upper()))

        return
    else:
        plt.plot(freq,np.log(spec),'b')
        plt.plot(n_freq,np.log(n_spec),'r')
        plt.plot([frmin,frmin],np.log([min(n_spec),max(spec)]),'g')
        plt.plot([frmax,frmax],np.log([min(n_spec),max(spec)]),'g')
        plt.plot(sitefreq,np.log(spec_x)-sitedata,color='skyblue',linestyle='--',linewidth=2)
        plt.plot(freq,np.log(synspec),'g--',linewidth=2)
##        plt.plot(fmfreq,np.log(fmspec),'y--',linewidth=2)
# =============================================================================
#         if POS.upper()=='S':
#             plt.plot(freq,np.log(synspec2),'k:',linewidth=2)
#             plt.plot(freq,np.log(synspec3),'k--',linewidth=2)
# =============================================================================
        plt.text(11,textx,'t* = %.2f' % invtstar)
        plt.text(11,textx-1,'1/Q = %.5f' % atten)
# =============================================================================
#         plt.text(10,9,'fitting1 = %.4f' % fitting)
#         plt.text(10,7.5,'fitting2 = %.4f' % fitting1)
#         plt.text(10,6,'fitting3 = %.4f' % fitting2)
#         plt.text(10,4.5,'fitting4 = %.4f' % fitting3)
# =============================================================================
# =============================================================================
#         plt.text(10,7.5,'curve fitting = %.4f' % resid)
# =============================================================================
# =============================================================================
#         plt.text(10,6,'misfit = %.4f' % saving[icase]['misfit'][ind])
#         plt.text(10,4.5,'lincorr = %.4f' % lincorr)
# =============================================================================
# =============================================================================
#         plt.text(10,3,'frange =[%.4f, %.4f]' % (frmin, frmax))
#         plt.text(10,1.5,'sd = %.4f' % saving[icase]['err'][ind])
# =============================================================================
        plt.xlim([0,xmax])
#        plt.ylim([ymin,ymax+3])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ln(A%s) on BH%s, nm/s' %(POS.lower(),chan[sch]))
        plt.title('%s  Station: %s' % (orid,sta))
        plt.savefig(g.figdir3+'/%s_%s_%sspectrum.pdf' % (orid,sta,POS.upper()))
        
        return
##    plt.plot(freqx,est,'k--')
##    plt.plot(freqx,dat,'k')

#   plt.ylim(np.log([min(n_spec),max(spec)]))


#    plt.ylim([-5,np.log(max(spec))])
# =============================================================================
#     plt.ylim([10**(int(np.log(min(min(spec),min(n_spec))))-2),
#                    10**(int(np.log(max(max(spec),max(n_spec))))+1)])
# =============================================================================



def fitting(saving,sta,orid,POS,lnmomen,fc,alpha,icase):
## CALCULATE HOW WELL THE SYNTHETIC SPECTRUM FITS THE DATA
## IF THE FITTING CURVE IS BELOW THE NOISE, THEN resid = 999999.
    if POS.upper()=='P':
        ind=0
        sch=0
    elif POS.upper()=='S':
        ind=1
        sch=saving['sch']
    else:
        raise ValueError("P or S wave?")
    corr=saving['corr'][ind]
    spec=saving['spec'][sch]
    freq=saving['freq'][sch]
    frmin=saving[2]['frmin'][ind]
    frmax=saving[2]['frmax'][ind]
    invtstar=saving[icase]['tstar'][ind]
##    print(invtstar)
    synspec=(corr*np.exp(lnmomen)*np.exp(-np.pi*freq*(freq**(-alpha))*invtstar)/(1+(freq/fc)**2))
    indx=np.all([(freq>=frmin),(freq<frmax)],axis=0)
    specx=spec[indx]
    freqx=freq[indx]
    synx=synspec[indx]
 ##   resid=np.linalg.norm(np.log(synx)-np.log(specx))/(len(freqx))
    resid=(1-((np.linalg.norm(np.log(synx)-np.log(specx)))**2/(len(freqx)-1)
             /np.var(np.log(specx))))    
    ##    RESID = 1-√(∑((ln(A(synthetic))-ln(A(observed)))^2))/(n-1)/σ(ln(A(observed))) Bill Menke, 'geophysical data analysis discrete inverse theory'
    resid1=(np.linalg.norm(np.log(synx)-np.log(specx))/(len(freqx)))/0.30
    ##    L2-NORM MISFIT = √(∑((ln(A(synthetic))-ln(A(observed)))^2))/n. Here 0.30 is just for normalization
    resid2=(np.linalg.norm(np.log(synx)-np.log(specx),ord=1)/(len(freqx)))/1.50
    ##    L1-NORM MISFIT = (∑|ln(A(synthetic))-ln(A(observed))|)/n. Here 1.50 is just for normalization
    resid3=(1-2*np.sum(np.log(synx)*np.log(specx))/(np.sum((np.log(specx))**2)+np.sum((np.log(synx))**2)))/0.8
    ##    CORRELATIVE FUNCTION. 1-2*∑(ln(A(synthetic))*ln(A(observed)))/∑((ln(A(synthetic))^2)+(ln(A(observed)))^2). Here 0.80 is just for normalization
# =============================================================================
#     resid1=(np.linalg.norm(np.log(synx)-np.log(specx))/(len(freqx)))
#     resid2=(np.linalg.norm(np.log(synx)-np.log(specx),ord=1)/(len(freqx)))
#     resid3=(1-2*np.sum(np.log(synx)*np.log(specx))/(np.sum((np.log(specx))**2)+np.sum((np.log(synx))**2)))
# =============================================================================

# =============================================================================
#     if resid1>1:
#         resid1=1
#     if resid2>1: 
#         resid2=1
#     if resid3>1:
#         resid3=1
# =============================================================================
 #   resid3=synx*specx
# =============================================================================
#     if sta == 'MALA':
#         print(len(specx))
#         print(ind)
#         print('synnnnnnnn11111111 %s'%(sta))
#         print(synx)
#         print('specccccc111111111 %s'%(sta))
#         print(specx)
# =============================================================================
#    print(resid3)
    df=abs(freq[1]-freq[0])
    nlowf=0
    narea=0
# =============================================================================
#     for ifreq in range(len(freq)):
#         if (freq[ifreq]>frmax and freq[ifreq]<19):
#             if (np.log(synspec[ifreq])<np.log(spec[ifreq]) or
#                  np.log(synspec[ifreq])>np.log(spec[ifreq])+1):
#                 narea=narea+np.log(spec[ifreq])-np.log(synspec[ifreq])
#             if np.log(synspec[ifreq])>np.log(spec[ifreq])+2:
#                 nlowf=nlowf+5
#             elif np.log(synspec[ifreq])>np.log(spec[ifreq])+1:
#                 nlowf=nlowf+1
#     if narea<-10 and nlowf*df>3:
#         resid=0
# =============================================================================
    return resid,resid1,resid2,resid3












def buildd(saving,stalst,fc,POS,icase,lnM=0):
## Build data matrix
##      d = [ln(A1)-ln(C1)+ln(1+(f1i/fc)**2),                   ##
##           ln(A2)-ln(C2)+ln(1+(f2i/fc)**2),                   ##
##           ln(AM)-ln(CM)+ln(1+(fMi/fc)**2)]                   ##
## INPUT:   saving - saved spectrum for each station: saving[sta][1]['p']
##          stalst - list of used stations
##          fc     - corner frequency
##          POS    - 'P' or 'S'
##          icase  - 1: high quality for finding best fc and alpha
##                   2: low quality for t* inversion
##                   3: low quality for t* inversion without bad fitting
##          lnM    - when POS='S', log of seismic moment
## OUTPUT:  data - data matrix for t* inversion
    if icase==3:
        icase=2
    if POS.upper()=='P':
        ind=0
    elif POS.upper()=='S':
        ind=1
    else:
        raise ValueError("P or S wave?")
    for ista in range(len(stalst)):
        sta=stalst[ista]
        freq_x = saving[sta][icase][POS.lower()][0]
        spec_x = saving[sta][icase][POS.lower()][1]
        correc = saving[sta]['corr'][ind]
        stad = np.array([np.log(spec_x)-np.log(correc)
                        +np.log(1+(freq_x/fc)**2)-lnM]).transpose()
##        print(sta,POS,max(stad),lnM)
        if ista==0:
            data=stad
        else:
            data=np.vstack((data,stad))
    return data



def buildG(saving,stalst,alpha,POS,icase,mw_flag):
## Build G matrix
##      G = [[1, -pi*f1i*f1i**(-alpha), 0, ..., 0],             ##
##           [1, 0, -pi*f2i*f2i**(-alpha), ..., 0],             ##
##           [1, 0, 0, ..., -pi*fMi*fMi**(-alpha)]]             ##
## INPUT:   saving - saved frequency for each station: saving[sta][1]['p']
##          stalst - list of used stations
##          alpha  - alpha value(s)
##          POS    - 'P' or 'S'
##          icase  - 1: high quality for finding best fc and alpha
##                   2: low quality for t* inversion
##                   3: low quality for t* inversion without bad fitting
## OUTPUT:  G - G matrix for t* inversion
    
# =============================================================================
# #    l = len(saving[sta][2]['p'][0])
# #    G = np.array([0]*(len(stalst)+1))
#     l = len(stalst)+1
#     G = np.array([0]*l)
#     print('come on')
# =============================================================================
    if icase==3:
        icase=2
    if POS.upper()=='P':
        ind=0
    elif POS.upper()=='S':
        ind=1
    else:
        raise ValueError("P or S wave?")
    for ista in range(len(stalst)):
        sta=stalst[ista]
        for alco in range(len(alpha)):
            freq_x = saving[sta][icase][POS.lower()][0]
            exponent = -1*np.pi*freq_x*(freq_x**(-alpha[alco]))
            exponent = np.array([exponent]).transpose()
            if alco==0:
                Gblock = np.atleast_3d(exponent)
            else:
                Gblock = np.dstack((Gpblock,exponent))
        if ista==0:
            G=Gblock
        else:
            oldblock=np.hstack((G,np.zeros((G.shape[0],1,len(alpha)))))
            newblock=np.hstack((np.zeros((Gblock.shape[0],G.shape[1],len(alpha))),Gblock))
            G=np.vstack((oldblock,newblock))
    if mw_flag == False:
        if POS.upper()=='P':
            G=np.hstack((np.ones((G.shape[0],1,len(alpha))),G))
    return G

