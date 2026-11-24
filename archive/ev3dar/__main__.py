import argparse
import json
from pathlib import Path
from .common import load_config,read_json,load_manifest,write_json


def main():
    parser=argparse.ArgumentParser(description='3DAR_EV portable baseline. Default training device is Spark CUDA.')
    parser.add_argument('command',choices=['prepare','features','train','evaluate','run','summarize'])
    parser.add_argument('--data-root',type=Path,help='Folder containing 측정_목록.csv and 측정_데이터/')
    parser.add_argument('--out',type=Path,default=Path('runs/baseline'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seed',type=int,choices=[42,43,44])
    parser.add_argument('--task',choices=['id','ssl','both'],default='both')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='CPU requires explicit selection; never used automatically.')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--no-maps',action='store_true')
    args=parser.parse_args()
    resolved=args.out/'config.resolved.json'
    cfg=load_config(args.config) if args.config or not resolved.exists() else read_json(resolved)
    if args.epochs is not None:cfg['epochs']=args.epochs
    if args.workers is not None:cfg['workers']=args.workers
    seeds=[args.seed] if args.seed is not None else cfg['seeds']
    if args.command in ['prepare','features','run'] and args.data_root is None:
        parser.error('--data-root must point to the separately supplied3DAR_EV folder.')
    if args.command in ['prepare','run']:
        from .data import prepare
        print(json.dumps(prepare(args.data_root,args.out,cfg),ensure_ascii=False,indent=2),flush=True)
    if args.command in ['features','run']:
        from .features import build_cache
        if not resolved.exists():parser.error('Run prepare first.')
        build_cache(args.data_root,args.out,cfg)
    if args.command in ['train','run','evaluate','summarize']:
        from .engine import train_task,evaluate,summarize
        if args.command in ['train','run']:
            write_json(resolved,cfg)
            for seed in seeds:
                for task in (['ssl','id'] if args.task=='both' else [args.task]):
                    train_task(args.out,cfg,seed,task,args.device)
        if args.command in ['evaluate','run']:
            if args.command=='run' and args.task!='both':
                print('One task trained; combined evaluation requires both checkpoints. Run evaluate after both.',flush=True)
            else:
                for seed in seeds:
                    result=evaluate(args.out,cfg,seed,args.device,not args.no_maps)
                    print(json.dumps({k:result[k] for k in ['seed','type_macro_f1','fault_f1','ssl_labels_constant']},ensure_ascii=False),flush=True)
        if args.command=='summarize' or (args.command=='run' and args.task=='both'):
            if all((args.out/f'seed{s}/test_metrics.json').exists() for s in cfg['seeds']):
                print(json.dumps(summarize(args.out,cfg['seeds']),ensure_ascii=False,indent=2))
    print('Done. Raw dataset was read only.',flush=True)


if __name__=='__main__':main()
