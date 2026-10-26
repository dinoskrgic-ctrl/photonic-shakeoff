import pickle,sys,numpy as np
from PIL import Image
rects,segs=pickle.load(open(sys.argv[1],'rb'))
cb=np.array(Image.open(sys.argv[2]).convert('RGB'),float)/255.  # 4x77 colorbar, assume top row=+1
cbcol=cb[:,1,:]; n=len(cbcol)
y0,y1=27.3446,105.979  # image placement
vals=[]
# value at image row r: y = y1 - (r+0.5)*(y1-y0)/n ; parity = -1 + (y-27.41)/(106.43-27.41)*2
cbv=np.array([-1+((y1-(r+0.5)*(y1-y0)/n)-27.41)/(106.43-27.41)*2 for r in range(n)])
cells=[]
for p0,p1,col in rects:
    if 49<p0[0]<224 and 26<p0[1]<108 and p1[0]-p0[0]>2 and len(col)==3:
        c=np.array(col); i=np.argmin(((cbcol-c)**2).sum(1)); err=((cbcol[i]-c)**2).sum()
        fp=7.2+((p0[0]+p1[0])/2-73.35)/208.675; t=((p0[1]+p1[1])/2-27.41)/25.46*100
        cells.append((fp,t,cbv[i],err,p1[1]-p0[1]))
cells=np.array(cells); print(len(cells),'max color err',cells[:,3].max())
np.savetxt(sys.argv[3],cells[:,:3],fmt='%.4f',header='Putterman et al. arXiv:2409.17556v2, App. Fig. (pump_loss) panel (c): storage parity vs pump frequency and delay, extracted from vector cells; colour mapped to parity via the figure colorbar (nearest colour, +-0.03).\ncol1 pump frequency (GHz, cell centre), col2 delay (us, cell centre), col3 parity')
# steady-state parity: mean over delay 200-300 us per pump column
fps=np.round(cells[:,0],3); out=[]
for f in np.unique(fps):
    m=(fps==f)&(cells[:,1]>200); out.append((f,cells[m,2].mean(),m.sum()))
out=np.array(out)
np.savetxt(sys.argv[4],out[:,:2],fmt='%.4f',header='Late-time (200-300 us) mean storage parity per pump frequency, from the same extraction. col1 pump GHz, col2 parity')
for f,p,k in out: print('%.3f %+.2f %s'%(f,p,'#'*int((1-p)*20)))
