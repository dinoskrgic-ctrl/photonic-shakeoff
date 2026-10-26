import pickle,re,sys,numpy as np,collections
out=pickle.load(open(sys.argv[1],'rb'))
s=max(out,key=lambda t:len(t[2]))[2].decode('latin1')
toks=s.split(); M=np.eye(3); st=[]; num=[]; col=None; cur=[]; rects=[]; segs=[]
ops=collections.Counter()
for t in toks:
    if re.fullmatch(r'-?\d*\.?\d+',t): num.append(float(t)); continue
    ops[t]+=1
    if t=='q': st.append((M.copy(),col))
    elif t=='Q': M,col=st.pop()
    elif t=='cm':
        a,b,c,d,e,f=num[-6:]; M=np.array([[a,b,0],[c,d,0],[e,f,1.]])@M
    elif t in('rg','sc','scn','k','g'): col=tuple(num)
    elif t=='re':
        x,y,w,h=num[-4:]; p0=np.array([x,y,1.])@M; p1=np.array([x+w,y+h,1.])@M; cur=[('re',p0,p1)]
    elif t=='m': cur=[('m',np.array([num[-2],num[-1],1.])@M)]
    elif t=='l': cur.append(('l',np.array([num[-2],num[-1],1.])@M))
    elif t in('f','f*','F'):
        if cur and cur[0][0]=='re': rects.append((cur[0][1],cur[0][2],col))
        elif cur:
            pts=np.array([c[1][:2] for c in cur]); rects.append((pts.min(0),pts.max(0),col))
        cur=[]
    elif t in('S','n','s'):
        if cur and cur[0][0]=='m': segs.append(np.array([c[1][:2] for c in cur]))
        cur=[]
    num=[]
print(ops.most_common(25))
print('filled',len(rects))
r=[x for x in rects if x[0][1]<130 and x[1][1]<130 and (x[1][0]-x[0][0])<10]
print('small rects low panel',len(r))
xs=sorted(set(round(x[0][0],2) for x in r)); ys=sorted(set(round(x[0][1],2) for x in r))
print(len(xs),xs[:5],xs[-3:]); print(len(ys),ys[:5],ys[-3:])
print(r[:3])
pickle.dump((rects,segs),open(sys.argv[2],'wb'))
