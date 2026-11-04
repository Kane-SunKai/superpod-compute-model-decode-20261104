import argparse
import json
from pathlib import Path
import sys
from . import GemmRequest, Hardware, Policy, Mapping, predict, evaluate_mapping
from .cache import MappingCache

def write_json(value,path):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def parser():
    p=argparse.ArgumentParser(description='Superpod independent single-GPU compute model (no GPU required)')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('gemm','evaluate'):
        c=sub.add_parser(name,help='Search mappings' if name=='gemm' else 'Evaluate one mapping and export its representative wave trace')
        c.add_argument('--m',type=int,required=True);c.add_argument('--n',type=int,required=True);c.add_argument('--k',type=int,required=True)
        c.add_argument('--dtype',choices=('fp16','bf16'),default='fp16');c.add_argument('--hardware',type=Path);c.add_argument('--policy',type=Path)
        c.add_argument('--backend',choices=('auto','array','peak','mmac','mac-throughput'),default='auto');c.add_argument('--output',type=Path)
        c.add_argument('--json',action='store_true',help='Print full JSON instead of a concise summary')
        if name=='gemm':
            c.add_argument('--cache',type=Path,default=Path('.cache/mappings.sqlite3'));c.add_argument('--no-cache',action='store_true');c.add_argument('--force',action='store_true');c.add_argument('--progress',action='store_true')
        else:
            c.add_argument('--mapping',required=True,help='bm,bn,bk,gm,gn,L2_buffers,shared_buffers[,output_chunk_bytes,output_buffers[,register_input_buffers]]')
            c.add_argument('--trace-panels',type=int,default=8,help='At most this many panels per representative wave, not a whole-kernel instruction trace')
    c=sub.add_parser('hardware',help='Print or export the bundled A100 profile');c.add_argument('--output',type=Path)
    c=sub.add_parser('cache',help='Inspect or clear the selected mapping database');c.add_argument('action',choices=('count','clear'));c.add_argument('--path',type=Path,default=Path('.cache/mappings.sqlite3'))
    sub.add_parser('operators',help='List implemented operators')
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='hardware':
            result=Hardware.load().to_dict()
            if args.output:write_json(result,args.output)
            print(json.dumps(result,indent=2));return 0
        if args.command=='operators':print('gemm: implemented\nsoftmax, layernorm: extension interfaces only; not implemented');return 0
        if args.command=='cache':
            cache=MappingCache(args.path)
            if args.action=='clear':cache.clear()
            print(f'entries={cache.count()}');return 0
        request=GemmRequest(args.m,args.n,args.k,args.dtype);hardware=Hardware.load(args.hardware)
        policy=Policy(**json.loads(args.policy.read_text(encoding='utf-8-sig'))) if args.policy else Policy()
        if args.command=='evaluate':
            values=[int(v) for v in args.mapping.split(',')]
            if len(values) not in (7,9,10):raise ValueError('--mapping requires 7, 9, or 10 integers (the 10th is register operand buffers)')
            if args.trace_panels<0 or args.trace_panels>10000:raise ValueError('--trace-panels must be 0..10000')
            result=evaluate_mapping(request,Mapping(*values),hardware,policy=policy,backend=args.backend,trace_limit=args.trace_panels)
            best=result
        else:
            progress=(lambda total,valid:print(f'candidates={total}, feasible={valid}',file=sys.stderr)) if args.progress else None
            result=predict(request,hardware,policy=policy,backend=args.backend,cache_path=None if args.no_cache else args.cache,force=args.force,progress=progress)
            best=result['best']
        if args.output:write_json(result,args.output)
        if args.json:print(json.dumps(result,indent=2,ensure_ascii=False))
        elif best:
            print(f"backend={args.backend} duration_us={best['duration_us']:.6f}")
            print('mapping='+json.dumps(best['mapping'],sort_keys=True))
            print(f"waves={best['wave_count']} HBM_input_MiB={best['hbm_read_bytes']/2**20:.6f} L2_to_Core_MiB={best['l2_to_core_bytes']/2**20:.6f}")
            if args.command=='gemm':print(f"cache_hit={result['cache_hit']} candidates={result['searched_candidates']} feasible={result['feasible_candidates']}")
            print('array_sources='+json.dumps(best['array_sources_wave_panels'],sort_keys=True))
            if policy.epilogue_s_per_wave==0:print('scope: output conversion/layout cost is zero/un-calibrated; model estimate, not GPU measurement')
        else:print(json.dumps(result['special'],indent=2))
        if args.output:print('saved='+str(args.output.resolve()),file=sys.stderr)
        return 0
    except (ValueError,TypeError,OSError,NotImplementedError,json.JSONDecodeError) as exc:
        print('error: '+str(exc),file=sys.stderr);return 2
