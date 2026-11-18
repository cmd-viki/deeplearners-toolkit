"""Prequential stream trainer: predict before learning each incoming chunk."""
import sys
import numpy as np
from .metrics import StreamingMetrics
class StreamTrainer:
    def __init__(self,model,positive_label=1):
        self.model=model; self.metrics=StreamingMetrics(positive_label); self.history=[]
    def fit_chunk(self,X,y):
        X=np.asarray(X); y=np.asarray(y)
        tree=self.model.steps[-1][1] if hasattr(self.model,'steps') else self.model
        try:
            if tree.root_ is None:
                raise RuntimeError('no fitted tree yet')
            prediction=self.model.predict(X)
            self.metrics.update(y,prediction)
            accuracy=float(np.mean(prediction==y))
        except RuntimeError:
            accuracy=np.nan # no model until first buffer flush
        self.model.partial_fit(X,y)
        tree=self.model.steps[-1][1] if hasattr(self.model,'steps') else self.model
        memory=int(sum(a.nbytes for a in getattr(tree,'_xb',[]))+sum(a.nbytes for a in getattr(tree,'_yb',[])))
        row={'chunk':len(self.history)+1,'accuracy':accuracy,'cumulative_accuracy':self.metrics.result()['accuracy'],
             'buffered_samples':tree.buffered_samples_,'retrain_count':tree.retrain_count_,'buffer_bytes':memory}
        self.history.append(row)
        return row
    def score_chunk(self,X,y):
        pred=self.model.predict(X)
        return float(np.mean(pred==np.asarray(y)))
