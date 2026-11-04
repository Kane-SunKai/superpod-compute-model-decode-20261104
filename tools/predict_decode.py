"""Single entry point for the current BW1100 FP16 Decode model."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from superpod_compute.decode import predict

def main():
    p=argparse.ArgumentParser(description='Predict contiguous FP16 A[M,K] @ B[K,N] on BW1100; 1<=M<=32, N/K multiples of64.')
    for name in ('m','n','k'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--output',type=pathlib.Path)
    args=p.parse_args();result=predict([args.m,args.n,args.k])
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"M/N/K={args.m}/{args.n}/{args.k}; tile={result['tile']}; predicted={result['duration_us']:.3f} us")
    print('MMAC=8 cycles; software prior + HBM/L2/Shared pipeline; '+result['model_version'])
if __name__=='__main__':main()
