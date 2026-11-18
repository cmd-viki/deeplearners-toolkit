"""Reusable plotting functions; returns matplotlib figure/axes."""
import numpy as np
import matplotlib.pyplot as plt

def _finish(fig,save_path=None,show=False):
    fig.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches='tight')
    if show: plt.show()
    return fig

def plot_metric_over_time(metric_values,title='Metric over time',ylabel='Metric',save_path=None,show=False):
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.arange(1,len(metric_values)+1),metric_values,marker='o')
    ax.set(xlabel='Chunk',ylabel=ylabel,title=title); ax.grid(alpha=.25)
    return _finish(fig,save_path,show)

def compare_models(metric1,metric2,labels=('Model A','Model B'),save_path=None,show=False):
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(metric1,label=labels[0]); ax.plot(metric2,label=labels[1]); ax.legend()
    ax.set(xlabel='Chunk',ylabel='Metric',title='Streaming model comparison'); ax.grid(alpha=.25)
    return _finish(fig,save_path,show)

def plot_predictions_vs_ground_truth(y_true,y_pred,save_path=None,show=False):
    y=np.asarray(y_true); p=np.asarray(y_pred)
    if len(y)!=len(p): raise ValueError('length mismatch')
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(y,'o',label='Truth',alpha=.65); ax.plot(p,'x',label='Prediction',alpha=.65)
    ax.set(xlabel='Sample',ylabel='Class',title='Predictions vs ground truth'); ax.legend()
    return _finish(fig,save_path,show)
