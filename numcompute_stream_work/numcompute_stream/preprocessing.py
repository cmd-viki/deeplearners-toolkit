"""Incremental feature transformers; numeric scalers ignore missing observations."""
import numpy as np

def _numeric(X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or not X.shape[0]: raise ValueError('expected nonempty 2D array')
    if np.isinf(X).any(): raise ValueError('infinite values not supported')
    return X

class StandardScaler:
    def __init__(self): self.n_samples_seen_=None
    def fit(self,X):
        self.n_samples_seen_=None
        return self.partial_fit(X)
    def partial_fit(self,X,y=None):
        X=_numeric(X); n=np.sum(~np.isnan(X),axis=0)
        sums=np.nansum(X,axis=0)
        mean=np.divide(sums,n,out=np.zeros(X.shape[1]),where=n>0)
        sq=np.nansum((X-mean)**2,axis=0)
        if self.n_samples_seen_ is None:
            self.n_samples_seen_=n; self.mean_=mean; self.M2_=sq
        else:
            if len(n)!=len(self.mean_): raise ValueError('feature count mismatch')
            old=self.n_samples_seen_; total=old+n; delta=mean-self.mean_
            self.mean_+=np.divide(delta*n,total,out=np.zeros_like(delta),where=total>0)
            self.M2_+=sq+np.divide(delta**2*old*n,total,out=np.zeros_like(delta),where=total>0)
            self.n_samples_seen_=total
        self.var_=np.divide(self.M2_,self.n_samples_seen_,out=np.zeros_like(self.M2_),where=self.n_samples_seen_>0)
        self.scale_=np.sqrt(self.var_); self.scale_[self.scale_==0]=1
        return self
    def transform(self,X):
        X=_numeric(X)
        if X.shape[1]!=len(self.mean_): raise ValueError('feature count mismatch')
        return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class Imputer:
    """Online per-feature mean imputation; all-missing features use zero."""
    def __init__(self): self.count_=None
    def fit(self,X): self.count_=None; return self.partial_fit(X)
    def partial_fit(self,X,y=None):
        X=_numeric(X); count=np.sum(~np.isnan(X),axis=0); sums=np.nansum(X,axis=0)
        if self.count_ is None: self.count_=count; self.sum_=sums
        else:
            if len(count)!=len(self.count_): raise ValueError('feature count mismatch')
            self.count_+=count; self.sum_+=sums
        self.statistics_=np.divide(self.sum_,self.count_,out=np.zeros_like(self.sum_),where=self.count_>0)
        return self
    def transform(self,X):
        X=_numeric(X).copy()
        if X.shape[1]!=len(self.statistics_): raise ValueError('feature count mismatch')
        ii=np.where(np.isnan(X)); X[ii]=self.statistics_[ii[1]]
        return X
    def fit_transform(self,X): return self.fit(X).transform(X)

class OneHotEncoder:
    """Expands categories across partial_fit calls; output width may grow."""
    def __init__(self): self.categories_=None
    def fit(self,X): self.categories_=None; return self.partial_fit(X)
    def partial_fit(self,X,y=None):
        X=np.asarray(X,dtype=str)
        if X.ndim!=2: raise ValueError('expected 2D array')
        if self.categories_ is None: self.categories_=[np.unique(c) for c in X.T]
        else:
            if X.shape[1]!=len(self.categories_): raise ValueError('feature count mismatch')
            self.categories_=[np.unique(np.concatenate((old,col))) for old,col in zip(self.categories_,X.T)]
        return self
    def transform(self,X):
        X=np.asarray(X,dtype=str)
        if X.ndim!=2 or X.shape[1]!=len(self.categories_): raise ValueError('feature count mismatch')
        return np.hstack([(X[:,j,None]==cats).astype(float) for j,cats in enumerate(self.categories_)])
    def fit_transform(self,X): return self.fit(X).transform(X)
