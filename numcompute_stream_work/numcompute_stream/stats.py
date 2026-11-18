"""Online moments, exact retained quantiles and histogram counts."""
import numpy as np
class StreamingStats:
    def __init__(self,bins=10,range=None,window=None):
        self.bins=bins; self.range=range; self.window=window; self.reset()
    def reset(self):
        self.n_=0; self.mean_=0.; self.M2_=0.; self._history=[]
        return self
    def update_stats(self,X_chunk):
        x=np.asarray(X_chunk,dtype=float).ravel(); x=x[np.isfinite(x)]
        if not len(x): return self
        n=len(x); mean=float(np.mean(x)); m2=float(np.sum((x-mean)**2))
        delta=mean-self.mean_; total=self.n_+n
        self.mean_+=delta*n/total; self.M2_+=m2+delta**2*self.n_*n/total
        self.n_=total; self._history.extend(x.tolist())
        if self.window is not None: self._history=self._history[-self.window:]
        return self
    update=update_stats
    def result(self): return {'count':self.n_,'mean':self.mean_,'variance':self.M2_/self.n_ if self.n_ else np.nan}
    def quantile(self,q):
        if not self._history: return np.nan
        return np.quantile(self._history,q)
    def histogram(self): return np.histogram(self._history,bins=self.bins,range=self.range)
