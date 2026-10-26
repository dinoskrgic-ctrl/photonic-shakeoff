import pickle,sys,numpy as np
rects,segs=pickle.load(open(sys.argv[1],'rb'))
for sgm in segs:
    if len(sgm)==2:
        (a,b),(c,d)=sgm
        if b>130 and max(abs(a-c),abs(b-d))<6 and abs(a-c)+abs(b-d)>2: print('seg',round(a,2),round(b,2),round(c,2),round(d,2))
long=[s for s in segs if len(s)>20]
for s in long: print('curve',len(s),s[:,0].min().round(2),s[:,0].max().round(2),s[:,1].min().round(2),s[:,1].max().round(2))
pickle.dump(long,open(sys.argv[2],'wb'))
