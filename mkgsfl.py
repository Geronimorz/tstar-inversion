## PROGRAM TO CREATE GEOMETRIC SPREADING FILES FOR P AND S
## Written by S. Wei, MARCH 2015

import os
import numpy as np
import globaldb as g
import momcalc

workdir='/Users/zhangyurong/tstar/GS'
namedir = '/Users/zhangyurong/tstar/data/Eventname'
if not os.path.isdir(workdir):
    os.mkdir(workdir)
    
    
oridlst = []
for name in open(namedir).readlines():
    name = name.strip('\n')
    if name is not None:
        oridlst.append(name)

for orid in oridlst:
##    print(orid)

    outpfln=workdir+'/pgsfile_%s.txt' % (orid)
    outsfln=workdir+'/sgsfile_%s.txt' % (orid)
    outpfile=open(outpfln,'w')
    outsfile=open(outsfln,'w')
    
    g.dbread(orid)
    
    dbev=g.dbass
    dbev=sorted(dbev.items(), key=lambda item:item[1])
    num =len(dbev)
##        print(num)
    
    inimomp=np.array([1.0e-12,5.0e-13,1.0e-13])
    inimoms=np.array([1.0e-12,5.0e-13,1.0e-13])
    for i in range(num):
 #       dbev.record=ii
        sta = dbev[i][0]
        print(sta)
        delta=dbev[i][1]
        print(delta)
        depth=float(g.dbdep)
        depth=max(depth,0.00)
##        print(sta,delta,depth)    
        for phs in ['P', 'S']:
            moment = momcalc.correct(delta, depth, phs)
# =============================================================================
#             cmd='%s -9. %f %f %s' %(momcmd,delta,depth,phs)
#             osout=os.popen(cmd).read()
#             moment=float(osout.split()[0])
# =============================================================================
            if phs=='P':
                if moment!=0:
                    inimomp[0]=inimomp[1]
                    inimomp[1]=inimomp[2]
                    inimomp[2]=moment
                if moment==0:
                    moment=np.mean(inimomp[np.nonzero(inimomp)])
            elif phs=='S' and g.dbja[sta][1]!=-1:
                if moment!=0:
                    inimoms[0]=inimoms[1]
                    inimoms[1]=inimoms[2]
                    inimoms[2]=moment
                if moment==0:
                    moment=np.mean(inimoms[np.nonzero(inimoms)])
            
            if phs == 'P':
                outpfile.write('%.8e %s\n' % (moment,sta))
                print(moment)
            elif phs == 'S' and g.dbja[sta][1]!=-1:
                outsfile.write('%.8e %s\n' % (moment,sta))

    outpfile.close()
    outsfile.close()
            