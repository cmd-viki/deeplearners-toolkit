"""Cumulative/rolling binary metrics with exact retained-score ROC AUC."""
from collections import deque
import numpy as np

class StreamingMetrics:
    def __init__(self,positive_label=1,window=None):
        if window is not None and window<1: raise ValueError('window must be positive')
        self.positive_label=positive_label; self.window=window; self.reset()
    def reset(self): self._data=deque(maxlen=self.window); self._scores=deque(maxlen=self.window); return self
    def update(self,y_true_chunk,y_pred_chunk,y_score_chunk=None):
        y=np.asarray(y_true_chunk).ravel(); pred=np.asarray(y_pred_chunk).ravel()
        if len(y)!=len(pred): raise ValueError('y_true and y_pred size mismatch')
        if y_score_chunk is not None and len(np.asarray(y_score_chunk).ravel())!=len(y): raise ValueError('score size mismatch')
        scores=[None]*len(y) if y_score_chunk is None else np.asarray(y_score_chunk,dtype=float).ravel()
        self._data.extend(zip(y.tolist(),pred.tolist()))
        self._scores.extend(scores)
        return self
    def result(self):
        if not self._data: return {'count':0,'accuracy':np.nan,'precision':np.nan,'recall':np.nan,'f1':np.nan,'error':np.nan,'confusion_matrix':np.zeros((2,2),dtype=int),'auc':np.nan}
        pairs=np.asarray(self._data); y=pairs[:,0]==self.positive_label; p=pairs[:,1]==self.positive_label
        tp=int(np.sum(y&p)); tn=int(np.sum(~y&~p)); fp=int(np.sum(~y&p)); fn=int(np.sum(y&~p))
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        accuracy=(tp+tn)/len(y); auc=np.nan
        scores=np.asarray(list(self._scores),dtype=object)
        if all(s is not None for s in scores) and np.any(y) and np.any(~y):
            s=np.asarray(scores,dtype=float)
            if np.all(np.isfinite(s)):
                order=np.argsort(s,kind='stable'); ranks=np.empty(len(s),dtype=float)
                sorted_s=s[order]; boundaries=np.r_[0,np.flatnonzero(np.diff(sorted_s)!=0)+1,len(s)]
                for a,b in zip(boundaries[:-1],boundaries[1:]): ranks[order[a:b]]=(a+b+1)/2
                npos=np.sum(y); nneg=len(y)-npos
                auc=float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))
        return {'count':len(y),'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,
                'error':1-accuracy,'confusion_matrix':np.array([[tn,fp],[fn,tp]]),'auc':auc}
