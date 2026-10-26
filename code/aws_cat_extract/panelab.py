import pickle,sys,numpy as np
from PIL import Image
rects,segs=pickle.load(open(sys.argv[1],'rb'))
D=sys.argv[4]
# panel a curves: any polyline in panel a region with >=5 pts
cur=[s for s in segs if len(s)>=5 and s[:,1].min()>250 and s[:,0].min()>50]
print('panel a curves',len(cur))
rows=[]
for k,s in enumerate(sorted(cur,key=lambda s:s[:,1].mean())):
    f=7.2+(s[:,0]-77.85)/197.5; b=2.96+(s[:,1]-267.27)/(22.21/0.02)
    # kinks: largest |db| between consecutive points
    d=np.abs(np.diff(b)); i=np.argsort(d)[-3:]
    print('curve',k,'npts',len(s),'mean fb %.4f'%b.mean(),'kinks at', [(round(f[j],3),round(d[j]*1e3,2)) for j in sorted(i)])
    for x,y in zip(f,b): rows.append((k,x,y))
np.savetxt(D+'/model_buffer_freq_vs_pump.txt',np.array(rows),fmt='%.5f',header='Putterman et al. App. Fig. pump_loss panel (a): their Floquet-model Stark-shifted buffer frequency vs pump frequency, curves for eps_p ascending in index order of mean frequency (index 0 = lowest buffer freq = largest eps_p 0.1). col1 curve index, col2 pump GHz, col3 buffer GHz')
# panel b image
for nm in sys.argv[2:4]:
    im=Image.open(nm); print(nm,im.size,im.mode)
img=np.array(Image.open(sys.argv[2]).convert('RGB'),float)/255.
cb=np.array(Image.open(sys.argv[3]).convert('RGB'),float)/255.; cbc=cb[:,1,:]; n=len(cbc)
y0,y1=136.5381,223.5632
cbv=np.array([-30+((y1-(r+0.5)*(y1-y0)/n)-147.07)/(221.9-147.07)*30 for r in range(n)])
H,W=img.shape[:2]
dB=np.zeros((H,W))
for r in range(H):
    for c in range(W):
        dB[r,c]=cbv[np.argmin(((cbc-img[r,c])**2).sum(1))]
# image placement x 51.4469..222.4613, y 139.4008..221.3664 ; row 0 = top
fx=7.2+((51.4469+(np.arange(W)+0.5)*(222.4613-51.4469)/W)-82.31)/181.55
fy=2.90+((221.3664-(np.arange(H)+0.5)*(221.3664-139.4008)/H)-149.28)/(27.23/0.05)
np.savez(D+'/buffer_S21_vs_pump.npz',pump_GHz=fx,probe_GHz=fy,S21_dB=dB)
mins=[(fx[c],fy[np.argmin(dB[:,c])],dB[:,c].min()) for c in range(W)]
np.savetxt(D+'/buffer_freq_measured_vs_pump.txt',np.array(mins),fmt='%.4f',header='Measured buffer resonance (S21 minimum) vs pump frequency, from panel (b) raster (eps_p ~0.08-0.1). col1 pump GHz, col2 probe GHz at minimum, col3 min S21 dB')
for c in range(0,W,6): print('%.3f %.4f %.1f'%mins[c])
