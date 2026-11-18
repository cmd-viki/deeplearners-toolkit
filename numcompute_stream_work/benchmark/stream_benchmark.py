"""Reproducible benchmark: vectorised vs loop split scan, and streaming buffer sizes."""
import csv
import time
from pathlib import Path
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier

def loop_gini_threshold(x,y):
    best=(-1,None)
    for threshold in np.unique(x)[:-1]:
        left=y[x<=threshold];right=y[x>threshold]
        def gini(a):
            if len(a)==0:return 0
            return 1-sum((sum(a==k)/len(a))**2 for k in np.unique(y))
        gain=1-sum((sum(y==k)/len(y))**2 for k in np.unique(y))-(len(left)*gini(left)+len(right)*gini(right))/len(y)
        if gain>best[0]:best=(gain,threshold)
    return best

def vector_gini_threshold(x,y):
    order=np.argsort(x); x=x[order]; y=y[order]
    cut=np.flatnonzero(np.diff(x)>0)
    counts=np.cumsum(np.column_stack((y==0,y==1)),axis=0)
    l=counts[cut]; r=counts[-1]-l; nl=l.sum(axis=1);nr=r.sum(axis=1)
    gain=1-np.sum((counts[-1]/len(y))**2)-(nl*(1-np.sum((l/nl[:,None])**2,axis=1))+nr*(1-np.sum((r/nr[:,None])**2,axis=1)))/len(y)
    i=np.argmax(gain);return float(gain[i]),float(x[cut[i]])

def run():
    rng=np.random.default_rng(42); x=rng.normal(size=1200);y=(x+rng.normal(size=1200)*.7>0).astype(int)
    timings={}
    for label,fn in [('loop',loop_gini_threshold),('vector',vector_gini_threshold)]:
        t=time.perf_counter();score,threshold=fn(x,y);timings[label]=time.perf_counter()-t
        print(label,'seconds',round(timings[label],6),'gain',round(score,5),'threshold',round(threshold,5))
    print('speedup',round(timings['loop']/timings['vector'],2),'x')
    X=rng.normal(size=(1200,3)); target=(X[:,0]+X[:,1]*.3>0).astype(int)
    records=[]
    for bs in (50,150,300,600):
        tree=DecisionTreeClassifier(max_depth=5,buffer_size=bs);correct=total=0
        start=time.perf_counter()
        for i in range(0,1000,50):
            xx=X[i:i+50];yy=target[i:i+50]
            if tree.root_ is not None:
                correct+=int(np.sum(tree.predict(xx)==yy));total+=len(yy)
            tree.partial_fit(xx,yy)
        elapsed=time.perf_counter()-start
        final_acc=tree.score(X[1000:],target[1000:])
        row=(bs,tree.retrain_count_,elapsed,correct/total if total else np.nan,final_acc)
        records.append(row);print('buffer/retrains/sec/prequential/final',row)
    with open(Path(__file__).with_name('results.csv'),'w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['buffer_size','retrains','total_train_seconds','prequential_accuracy','holdout_accuracy']);writer.writerows(records)
if __name__=='__main__':run()
