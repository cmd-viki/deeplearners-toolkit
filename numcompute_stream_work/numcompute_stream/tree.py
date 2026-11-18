"""From-scratch CART classification tree with periodic buffered retraining."""
from dataclasses import dataclass
import numpy as np

@dataclass
class _Node:
    prediction: object
    feature: int = -1
    threshold: float = 0.
    left: object = None
    right: object = None
    nan_left: bool = True

class DecisionTreeClassifier:
    """Binary CART tree; deterministic tie-breaking; NaNs routed to learned child.

    partial_fit buffers incoming chunks. When buffer_size is reached, a fresh
    tree is trained on *all* currently buffered samples and the buffer clears.
    The previous tree is used for prediction between retrains. A single chunk
    larger than buffer_size triggers one retrain on the whole chunk.
    """
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion='gini', min_impurity_decrease=1e-12):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError('max_depth must be None or a nonnegative integer')
        if min_samples_split < 2 or buffer_size < 1:
            raise ValueError('min_samples_split >= 2 and buffer_size >= 1 required')
        if criterion not in ('gini','entropy'): raise ValueError('criterion must be gini or entropy')
        self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.max_features=max_features; self.buffer_size=buffer_size
        self.criterion=criterion; self.min_impurity_decrease=min_impurity_decrease
        self._xb=[]; self._yb=[]; self.buffered_samples_=0; self.retrain_count_=0
        self.root_=None; self.n_features_in_=None
    def _validate(self,X,y=None):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[0]==0 or X.shape[1]==0: raise ValueError('X must be nonempty 2D')
        if np.isinf(X).any(): raise ValueError('infinite feature values not supported')
        if self.n_features_in_ is not None and X.shape[1]!=self.n_features_in_:
            raise ValueError('feature count mismatch')
        if y is None: return X
        y=np.asarray(y)
        if y.ndim!=1 or len(y)!=len(X): raise ValueError('y must be 1D with one label per row')
        return X,y
    def _impurity(self,counts):
        counts=np.asarray(counts,dtype=float)
        n=counts.sum(axis=-1,keepdims=True)
        prob=np.divide(counts,n,out=np.zeros_like(counts),where=n>0)
        if self.criterion=='gini': return 1-np.sum(prob**2,axis=-1)
        with np.errstate(divide='ignore',invalid='ignore'):
            logs=np.log2(prob,out=np.zeros_like(prob),where=prob>0)
        return -np.sum(prob*logs,axis=-1)
    def fit(self,X,y):
        X,y=self._validate(X,y)
        self.n_features_in_=X.shape[1]
        self.classes_,encoded=np.unique(y,return_inverse=True)
        self.n_classes_=len(self.classes_)
        self.root_=self._grow(X,encoded,0)
        return self
    def _grow(self,X,y,depth):
        counts=np.bincount(y,minlength=self.n_classes_)
        node=_Node(self.classes_[np.argmax(counts)])
        if (len(y)<self.min_samples_split or np.count_nonzero(counts)<=1 or
            (self.max_depth is not None and depth>=self.max_depth)): return node
        parent_imp=float(self._impurity(counts)); best_gain=self.min_impurity_decrease
        best=None
        n_features=X.shape[1]
        if self.max_features is None: features=np.arange(n_features)
        elif isinstance(self.max_features,int) and 1<=self.max_features<=n_features:
            features=np.arange(self.max_features) # deterministic subset
        else: raise ValueError('max_features must be None or integer in [1,n_features]')
        for f in features:
            col=X[:,f]; finite=np.isfinite(col); order=np.argsort(col[finite],kind='stable')
            vals=col[finite][order]; labels=y[finite][order]
            if len(vals)<2: continue
            candidates=np.flatnonzero(vals[1:]>vals[:-1])
            if len(candidates)==0: continue
            onehot=np.eye(self.n_classes_,dtype=np.int64)[labels]
            cumulative=np.cumsum(onehot,axis=0)
            left_base=cumulative[candidates]; total=cumulative[-1]
            nan_counts=counts-total
            for nan_left in (True,False):
                left=left_base+(nan_counts if nan_left else 0)
                right=counts-left
                nl=left.sum(axis=1); nr=right.sum(axis=1)
                valid=(nl>0)&(nr>0)
                gains=parent_imp-(nl*self._impurity(left)+nr*self._impurity(right))/len(y)
                gains[~valid]=-np.inf
                idx=int(np.argmax(gains)); gain=float(gains[idx])
                if gain>best_gain+1e-14:
                    cut=int(candidates[idx]); threshold=float(vals[cut]+(vals[cut+1]-vals[cut])/2)
                    if not np.isfinite(threshold): threshold=float(vals[cut])
                    best_gain=gain; best=(int(f),threshold,nan_left)
        if best is None: return node
        f,threshold,nan_left=best
        mask=(X[:,f]<=threshold)&np.isfinite(X[:,f])
        if nan_left: mask|=np.isnan(X[:,f])
        if mask.all() or (~mask).all(): return node
        node.feature=f; node.threshold=threshold; node.nan_left=nan_left
        node.left=self._grow(X[mask],y[mask],depth+1)
        node.right=self._grow(X[~mask],y[~mask],depth+1)
        return node
    def predict(self,X):
        X=self._validate(X)
        if self.root_ is None: raise RuntimeError('tree is not trained yet; buffer has not filled')
        result=np.empty(len(X),dtype=self.classes_.dtype)
        def visit(node,idx):
            if not len(idx): return
            if node.feature<0: result[idx]=node.prediction; return
            vals=X[idx,node.feature]
            left=(vals<=node.threshold)&np.isfinite(vals)
            if node.nan_left: left|=np.isnan(vals)
            visit(node.left,idx[left]); visit(node.right,idx[~left])
        visit(self.root_,np.arange(len(X)))
        return result
    def partial_fit(self,X_chunk,y_chunk):
        X,y=self._validate(X_chunk,y_chunk)
        if self.n_features_in_ is None: self.n_features_in_=X.shape[1]
        self._xb.append(X.copy()); self._yb.append(y.copy())
        self.buffered_samples_+=len(y)
        if self.buffered_samples_>=self.buffer_size:
            self.fit(np.concatenate(self._xb),np.concatenate(self._yb))
            self._xb.clear(); self._yb.clear(); self.buffered_samples_=0
            self.retrain_count_+=1
        return self
    def score(self,X,y):
        y=np.asarray(y)
        return float(np.mean(self.predict(X)==y))
