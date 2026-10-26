import pickle,sys,numpy as np
paths=pickle.load(open(sys.argv[1],'rb'))
p=np.array(max(paths,key=len))
f=2+(p[:,0]-53.55)/(36.645); dB=-80+(p[:,1]-92.71)/(16.917/20)
o=np.argsort(f,kind='stable'); f,dB=f[o],dB[o]
np.savetxt(sys.argv[2],np.c_[f,dB],fmt='%.5f',header='Putterman et al. arXiv:2409.17556v2, Fig.2(b) (scfigure_coherence.pdf) metamaterial filter S21, extracted from vector path.\ncol1 frequency (GHz), col2 S21 (dB, uncalibrated; convolved with 2.5-4 GHz isolator response per caption). Values below -80 dB were clipped in the plot.')
m=(f>2.5)&(f<3.7)
ff,dd=f[m],dB[m]
# local maxima
from numpy import convolve
k=np.ones(5)/5; ds=np.convolve(dd,k,'same')
pk=[i for i in range(3,len(ds)-3) if ds[i]==max(ds[i-3:i+4]) and ds[i]>-45]
for i in pk: print('peak %.3f GHz %.1f dB'%(ff[i],ds[i]))
# -3 dB / -10 dB edges relative to passband typical level
lvl=np.median(dd[(ff>2.7)&(ff<3.4)]); print('median passband level',lvl)
above=ff[dd>lvl-10]; print('edges (lvl-10 dB):',above.min(),above.max())
above=ff[dd>-50]; print('edges (-50 dB):',above.min(),above.max())
