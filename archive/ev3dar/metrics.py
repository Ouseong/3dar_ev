import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             roc_auc_score, average_precision_score)


def gaussian_target(degrees,sigma=5.0):
    x=np.arange(360);target=np.zeros(360,dtype=np.float32)
    for angle in degrees:
        d=np.abs(x-(float(angle)%360));d=np.minimum(d,360-d)
        target=np.maximum(target,np.exp(-.5*(d/sigma)**2))
    return target


def circular_peaks(prob,count,min_spacing=30):
    """Three periods make0/359 genuine interior candidates, not lost endpoints."""
    prob=np.asarray(prob,dtype=float)
    if prob.shape!=(360,):raise ValueError('Expected360 bins.')
    peaks,_=find_peaks(np.tile(prob,3),prominence=.01)
    candidates=[int(i-360) for i in peaks if 360<=i<720]
    candidates.sort(key=lambda i:(-prob[i],i));selected=[]
    for candidate in candidates:
        if all(min(abs(candidate-p),360-abs(candidate-p))>=min_spacing for p in selected):
            selected.append(candidate)
        if len(selected)>=count:break
    return selected


def angular_error(predicted,truth):
    truth=np.asarray(truth,dtype=float);predicted=np.asarray(predicted,dtype=float)
    if len(truth)==0:return 0.0
    if len(predicted)==0:return 180.0
    cost=np.abs(truth[:,None]-predicted[None,:]);cost=np.minimum(cost,360-cost)
    i,j=linear_sum_assignment(cost)
    # Do not make MAE artificially small by dropping undetected sources.
    return float((cost[i,j].sum()+180.0*(len(truth)-len(i)))/len(truth))


def classification_metrics(labels,probabilities,classes,threshold=.5):
    labels=np.asarray(labels,dtype=int).reshape(-1)
    probabilities=np.asarray(probabilities,dtype=float).reshape(-1,len(classes))
    predictions=probabilities.argmax(-1)
    fault=(labels!=0).astype(int);scores=1-probabilities[:,0]
    binary=(scores>=threshold).astype(int)
    both=len(np.unique(fault))==2
    return dict(n_regions=len(labels),type_accuracy=float(np.mean(predictions==labels)),
        type_macro_f1=float(f1_score(labels,predictions,labels=np.arange(len(classes)),average='macro',zero_division=0)),
        type_confusion_matrix=confusion_matrix(labels,predictions,labels=np.arange(len(classes))).tolist(),
        classes=classes,fault_accuracy=float(np.mean(binary==fault)),
        fault_f1=float(f1_score(fault,binary,zero_division=0)),
        fault_balanced_accuracy=float(balanced_accuracy_score(fault,binary)) if both else float(np.mean(binary==fault)),
        fault_roc_auc=float(roc_auc_score(fault,scores)) if both else None,
        fault_average_precision=float(average_precision_score(fault,scores)) if both else None,
        fault_confusion_matrix=confusion_matrix(fault,binary,labels=[0,1]).tolist())


def spatial_map(doa,region_prob,cfg):
    out=np.zeros((360,len(cfg['classes'])),dtype=np.float32)
    theta=np.arange(360)
    for i,pos in enumerate(cfg['positions']):
        a,b=cfg['rois'][pos]
        selected=((theta>=a)&(theta<b)) if b<=360 else ((theta>=a)|(theta<b-360))
        out[selected]=np.asarray(doa)[selected,None]*np.asarray(region_prob)[i]
    return out
