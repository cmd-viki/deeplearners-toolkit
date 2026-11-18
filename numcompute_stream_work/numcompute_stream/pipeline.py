"""Incremental sequential preprocessing and buffered tree pipeline."""
class Pipeline:
    def __init__(self,steps):
        if not steps or len(set(name for name,_ in steps))!=len(steps): raise ValueError('steps must be nonempty with unique names')
        self.steps=steps
    def partial_fit(self,X,y):
        for _,transformer in self.steps[:-1]:
            transformer.partial_fit(X)
            X=transformer.transform(X)
        self.steps[-1][1].partial_fit(X,y)
        return self
    def fit(self,X,y):
        for _,transformer in self.steps[:-1]: X=transformer.fit_transform(X)
        self.steps[-1][1].fit(X,y)
        return self
    def predict(self,X):
        for _,transformer in self.steps[:-1]: X=transformer.transform(X)
        return self.steps[-1][1].predict(X)
    def score(self,X,y):
        import numpy as np
        return float(np.mean(self.predict(X)==y))
