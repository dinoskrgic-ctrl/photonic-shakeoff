import pickle,re,sys
import numpy as np
out=pickle.load(open(sys.argv[1],'rb'))
big=max(out,key=lambda t:len(t[2]))[2].decode('latin1')
toks=big.split()
M=np.eye(3); stack=[]
paths=[];cur=[];num=[]
def tr(x,y):
    v=np.array([x,y,1.0])@M; return (v[0],v[1])
for t in toks:
    if re.fullmatch(r'-?\d*\.?\d+',t): num.append(float(t)); continue
    if t=='q': stack.append(M.copy())
    elif t=='Q': M=stack.pop() if stack else np.eye(3)
    elif t=='cm':
        a,b,c,d,e,f=num[-6:]; C=np.array([[a,b,0],[c,d,0],[e,f,1.0]]); M=C@M
    elif t=='m':
        if len(cur)>1: paths.append(cur)
        cur=[tr(num[-2],num[-1])]
    elif t=='l': cur.append(tr(num[-2],num[-1]))
    elif t=='c': cur.append(tr(num[-2],num[-1]))
    elif t=='re':
        x,y,w,h=num[-4:]; paths.append([tr(x,y),tr(x+w,y),tr(x+w,y+h),tr(x,y+h)])
    elif t in('S','s','f','F','f*','n','B','b','B*','b*'):
        if len(cur)>1: paths.append(cur)
        cur=[]
    num=[]
pickle.dump(paths,open(sys.argv[2],'wb'))
paths2=sorted(paths,key=len,reverse=True)
p=paths2[0]; xs=[a for a,b in p]; ys=[b for a,b in p]
print('trace',len(p),min(xs),max(xs),min(ys),max(ys))
# tick-like short segments near trace bbox
X0,X1,Y0,Y1=min(xs),max(xs),min(ys),max(ys)
for q in paths:
    if len(q)==2:
        (a,b),(c,d)=q
        if X0-15<a<X1+15 and Y0-25<b<Y1+15 and (abs(a-c)<0.01 or abs(b-d)<0.01) and max(abs(a-c),abs(b-d))<6:
            print('tick',round(a,2),round(b,2),round(c,2),round(d,2))
