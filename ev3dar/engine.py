"""Training/evaluation entrypoints. Never invoked by import or preparation."""
from pathlib import Path
import json,random,time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from .common import (load_manifest,records_by_ids,read_json,write_json,write_csv,
                     config_hash,digest)
from .features import cache_path,feature_key,fit_normalization
from .models import LocalizationCNN,SoundTypeCNN,focal_loss
from .metrics import (gaussian_target,circular_peaks,angular_error,
                      classification_metrics,spatial_map)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


class GpuThrottle:
    """Heat limiter: after each step, sleep in proportion to the step's busy time so the
    average GPU duty is about `duty` (0.5 -> ~50%, training ~2x slower). Results unchanged.
    Set env EV3DAR_GPU_DUTY=1 to disable. GB10 has no nvidia-smi power limit, hence software."""
    def __init__(self,duty=None):
        import os
        self.duty=float(os.environ.get('EV3DAR_GPU_DUTY',0.5) if duty is None else duty)
        self.t=time.perf_counter()
    def reset(self):self.t=time.perf_counter()
    def step(self):
        if self.duty>=1:return
        if torch.cuda.is_available():torch.cuda.synchronize()
        busy=time.perf_counter()-self.t
        time.sleep(busy*(1-self.duty)/max(self.duty,0.05));self.t=time.perf_counter()


class CacheDataset(Dataset):
    def __init__(self,records,out,cfg,norm):self.records=records;self.out=out;self.cfg=cfg;self.norm=norm
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with np.load(cache_path(self.out,r['uid']),allow_pickle=False) as z:
            if str(z['feature_key'].item())!=feature_key(r,self.cfg):
                raise ValueError('Stale feature cache: rerun features with the resolved configuration.')
            ssl=z['ssl'].astype(np.float32);views=z['id_views'].astype(np.float32)
            masks=z['id_masks'].copy();durations=z['view_durations'].copy();rpm=z['rpm'].copy()
        sm=np.asarray(self.norm['ssl']['mean'],np.float32)[:,None]
        ss=np.asarray(self.norm['ssl']['std'],np.float32)[:,None]
        im=np.asarray(self.norm['id']['mean'],np.float32)[None,None,:,None]
        istd=np.asarray(self.norm['id']['std'],np.float32)[None,None,:,None]
        ssl=(ssl-sm)/ss;views=((views-im)/istd)*masks[:,:,None,:]
        labels=np.array([self.cfg['classes'].index(c) for c in r['classes']],dtype=np.int64)
        target=gaussian_target([self.cfg['geometry'][p] for p in self.cfg['positions']],self.cfg['gaussian_sigma_degrees'])
        return dict(uid=r['uid'],ssl=torch.from_numpy(ssl),views=torch.from_numpy(views),
            masks=torch.from_numpy(masks),durations=torch.from_numpy(durations),rpm=torch.from_numpy(rpm),
            labels=torch.from_numpy(labels),count=torch.tensor(r['source_count']-1,dtype=torch.long),
            target=torch.from_numpy(target))


def make_loader(records,out,cfg,norm,shuffle=False,seed=42):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(CacheDataset(records,out,cfg,norm),batch_size=cfg['batch_size'],shuffle=shuffle,
        num_workers=cfg['workers'],generator=generator,pin_memory=torch.cuda.is_available())


def network(task,cfg):
    if task=='ssl':return LocalizationCNN(cfg['n_mels']*(2 if cfg['ssl_context_seconds']>0 else 1))
    return SoundTypeCNN(len(cfg['classes']),cfg['use_rpm'])


def forward(model,task,batch,device):
    if task=='ssl':return model(batch['ssl'].to(device,non_blocking=True))
    return model(*(batch[k].to(device,non_blocking=True) for k in ['views','masks','durations','rpm']))


def validate(model,task,loader,cfg,device):
    model.eval();losses=[];labels=[];probabilities=[]
    with torch.no_grad():
        for batch in loader:
            a,b=forward(model,task,batch,device)
            if task=='ssl':
                loss=F.cross_entropy(a,batch['count'].to(device))+cfg['location_loss_weight']*focal_loss(b,batch['target'].to(device))
                losses.append(float(loss)*len(batch['uid']))
            else:
                labels.append(batch['labels'].numpy());probabilities.append(a.softmax(-1).cpu().numpy())
    if task=='ssl':return dict(loss=sum(losses)/len(loader.dataset)), -sum(losses)/len(loader.dataset)
    metrics=classification_metrics(np.concatenate(labels),np.concatenate(probabilities),cfg['classes'],cfg['fault_threshold'])
    return metrics,metrics['type_macro_f1']


def train_task(out,cfg,seed,task,device='cuda'):
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run training in the Spark CUDA PyTorch environment; no automatic CPU fallback.')
    seed_all(seed);device=torch.device(device)
    records=load_manifest(out);split=read_json(Path(out)/f'splits/seed{seed}.json')
    manifest_digest=digest(Path(out)/'manifest.jsonl')
    if split['mode']!='recording_holdout_closed_source':raise ValueError('Unsupported split protocol.')
    train_records=records_by_ids(records,split['train']);valid_records=records_by_ids(records,split['validation'])
    run=Path(out)/f'seed{seed}';run.mkdir(exist_ok=True)
    normfile=run/'normalization.json'
    if normfile.exists():
        norm=read_json(normfile)
        if (norm.get('config_hash')!=config_hash(cfg) or norm['fitted_recording_ids']!=split['train']
                or norm.get('manifest_sha256')!=manifest_digest):
            raise ValueError('Normalization belongs to another config/split; use a fresh output directory.')
    else:
        norm=fit_normalization(train_records,out,cfg);norm.update(config_hash=config_hash(cfg),manifest_sha256=manifest_digest);write_json(normfile,norm)
    training=make_loader(train_records,out,cfg,norm,True,seed)
    validation=make_loader(valid_records,out,cfg,norm)
    model=network(task,cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=-float('inf');stale=0;history=[];throttle=GpuThrottle()
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.0;examples=0;tick=time.monotonic();throttle.reset()
        for batch in training:
            optimizer.zero_grad(set_to_none=True);a,b=forward(model,task,batch,device)
            if task=='ssl':
                loss=F.cross_entropy(a,batch['count'].to(device))+cfg['location_loss_weight']*focal_loss(b,batch['target'].to(device))
            else:loss=F.cross_entropy(a.reshape(-1,len(cfg['classes'])),batch['labels'].to(device).reshape(-1))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; checkpoint not updated.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=float(loss.detach())*len(batch['uid']);examples+=len(batch['uid']);throttle.step()
        metrics,score=validate(model,task,validation,cfg,device)
        row=dict(epoch=epoch,train_loss=total/examples,validation=metrics,selection_score=score,seconds=time.monotonic()-tick)
        history.append(row);print(json.dumps(dict(seed=seed,task=task,**row)),flush=True)
        if score>best:
            best=score;stale=0
            # Serialize immediately, avoiding the original state_dict reference bug.
            checkpoint=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                task=task,config=cfg,config_hash=config_hash(cfg),normalization=norm,seed=seed,epoch=epoch,
                validation=metrics,split=split,training_claim='closed-source held-out-recording baseline')
            checkpoint['manifest_sha256']=manifest_digest
            temp=run/(task+'_best.pending.pt');torch.save(checkpoint,temp);temp.replace(run/(task+'_best.pt'))
        else:stale+=1
        write_json(run/(task+'_history.json'),history)
        if stale>=cfg['patience']:break
    del model,optimizer
    if device.type=='cuda':torch.cuda.empty_cache()


def load_network(path,task,device):
    # Only load checkpoints produced by this project, not untrusted pickle files.
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt['task']!=task:raise ValueError('Checkpoint task mismatch.')
    model=network(task,ckpt['config']);model.load_state_dict(ckpt['state_dict']);model.to(device).eval()
    return model,ckpt


def evaluate(out,cfg,seed,device='cuda',save_maps=True):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable on this host.')
    device=torch.device(device);run=Path(out)/f'seed{seed}'
    id_model,ic=load_network(run/'id_best.pt','id',device)
    ssl_model,sc=load_network(run/'ssl_best.pt','ssl',device)
    if ic['config_hash']!=sc['config_hash'] or ic['split']!=sc['split']:
        raise ValueError('SSL/ID checkpoints must share exactly the same config and recording split.')
    if ic['config_hash']!=config_hash(cfg):raise ValueError('Use the config stored for these checkpoints.')
    if (ic['manifest_sha256']!=sc['manifest_sha256'] or
            ic['manifest_sha256']!=digest(Path(out)/'manifest.jsonl')):
        raise ValueError('Manifest changed after training. Use the original manifest/checkpoints.')
    all_records=load_manifest(out)
    records=records_by_ids(all_records,ic['split']['test']);lookup={r['uid']:r for r in records}
    loader=make_loader(records,out,cfg,ic['normalization']);labels=[];probs=[];rows=[]
    maes=[];count_correct=0;layout=list(cfg['geometry'].values());perpos={p:([],[]) for p in cfg['positions']}
    if save_maps:(run/'spatial_maps').mkdir(exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            il,_=forward(id_model,'id',batch,device);counts,loc=forward(ssl_model,'ssl',batch,device)
            ip=il.softmax(-1).cpu().numpy();dp=loc.sigmoid().cpu().numpy();predn=counts.argmax(-1).cpu().numpy()+1
            yy=batch['labels'].numpy();labels.append(yy);probs.append(ip)
            for j,uid in enumerate(batch['uid']):
                peaks=circular_peaks(dp[j],int(predn[j]),cfg['minimum_peak_spacing_degrees'])
                maes.append(angular_error(peaks,layout));count_correct+=int(predn[j]==3)
                if save_maps:
                    np.savez_compressed(run/'spatial_maps'/(uid+'.npz'),doa_probability=dp[j],
                        region_type_probability=ip[j],joint_map=spatial_map(dp[j],ip[j],cfg),
                        predicted_angles=np.asarray(peaks,dtype=np.float32))
                for k,pos in enumerate(cfg['positions']):
                    a,b=cfg['rois'][pos]
                    assigned=[p for p in peaks if (a<=p<b if b<=360 else p>=a or p<b-360)]
                    selected=max(assigned,key=lambda p:dp[j,p]) if assigned else None
                    label=int(ip[j,k].argmax());fault_prob=float(1-ip[j,k,0])
                    rec=dict(recording_id=uid,seed=seed,position=pos,predicted_type=cfg['classes'][label],
                        predicted_fault=int(fault_prob>=cfg['fault_threshold']),fault_probability=fault_prob,
                        predicted_angle_deg=selected,predicted_source_count=int(predn[j]),
                        gt_type=lookup[uid]['classes'][k],gt_fault=int(yy[j,k]!=0),
                        gt_reference_angle_deg=cfg['geometry'][pos],wav=lookup[uid]['wav'],band=lookup[uid]['band'])
                    rec.update({'prob_'+c:float(ip[j,k,ci]) for ci,c in enumerate(cfg['classes'])});rows.append(rec)
                    perpos[pos][0].append(yy[j,k]);perpos[pos][1].append(ip[j,k])
    truth=np.concatenate(labels)
    metrics=classification_metrics(truth,np.concatenate(probs),cfg['classes'],cfg['fault_threshold'])
    # Training-only position prior: quantify what the fixed layout/class balance
    # already predicts without listening to any audio. Never fit this on test.
    if 'train' in ic['split']:
        training_records=records_by_ids(all_records,ic['split']['train'])
        prior=np.ones((len(cfg['positions']),len(cfg['classes'])),np.float64)
        for r in training_records:
            for k,c in enumerate(r['classes']):prior[k,cfg['classes'].index(c)]+=1
        prior/=prior.sum(-1,keepdims=True)
        prior_predictions=np.broadcast_to(prior,(len(records),*prior.shape))
        metrics['training_position_prior_baseline']=classification_metrics(
            truth,prior_predictions,cfg['classes'],cfg['fault_threshold'])
    metrics.update(seed=seed,n_recordings=len(records),split_mode=ic['split']['mode'],
        per_position={p:classification_metrics(v[0],v[1],cfg['classes'],cfg['fault_threshold']) for p,v in perpos.items()},
        ssl_count_accuracy=count_correct/len(records),ssl_reference_layout_mae_degrees=float(np.mean(maes)),
        fixed_three_source_baseline_count_accuracy=1.0,fixed_layout_baseline_mae_degrees=0.0,
        ssl_labels_constant=True,geometry_status=cfg['geometry_status'],
        source_file_overlap=ic['split']['source_file_overlap'],
        generalization_claim='Known replay-source recordings only; NOT unseen-source or arbitrary-source-count localization.')
    write_json(run/'test_metrics.json',metrics);write_csv(run/'test_predictions.csv',rows)
    write_csv(run/'type_confusion_matrix.csv',[dict(actual=c,**{p:metrics['type_confusion_matrix'][i][j]
        for j,p in enumerate(cfg['classes'])}) for i,c in enumerate(cfg['classes'])])
    return metrics


def summarize(out,seeds):
    metrics=[read_json(Path(out)/f'seed{s}/test_metrics.json') for s in seeds]
    keys=['type_accuracy','type_macro_f1','fault_f1','fault_balanced_accuracy','fault_roc_auc',
          'ssl_count_accuracy','ssl_reference_layout_mae_degrees']
    result=dict(seeds=seeds,protocol='recording_holdout_closed_source',
        summary={k:dict(mean=float(np.mean([m[k] for m in metrics])),
                        std=float(np.std([m[k] for m in metrics],ddof=1)) if len(metrics)>1 else 0.0)
                 for k in keys if all(m[k] is not None for m in metrics)},
        warning='No unseen-source generalization claim. SSL count/layout labels are constant in this dataset.')
    write_json(Path(out)/'three_seed_summary.json',result);return result
