import re,zlib,sys,pickle
d=open(sys.argv[1],'rb').read()
out=[]
for m in re.finditer(rb'(\d+) 0 obj\s*<<(.*?)>>\s*stream\r?\n',d,re.S):
    hdr=m.group(2); L=re.search(rb'/Length (\d+)',hdr)
    if not L: continue
    s=m.end(); raw=d[s:s+int(L.group(1))]
    try: dec=zlib.decompress(raw) if b'Flate' in hdr else raw
    except Exception: dec=b''
    out.append((int(m.group(1)),hdr[:160],dec))
for n,h,dd in sorted(out,key=lambda t:-len(t[2]))[:12]: print(n,len(dd),h.replace(b'\n',b' '))
pickle.dump(out,open(sys.argv[2],'wb'))
