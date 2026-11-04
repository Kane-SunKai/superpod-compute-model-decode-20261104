"""BW1100 Decode: software-template selection and constrained service pipelines.

Physical MMAC issue remains eight cycles. The two service rates are calibrated
on development data, not claimed to be silicon peak rates. No validation timing
is consulted by this predictor.
"""
import json,pathlib
from .contracts import Hardware,Policy,Mapping
from .operators import GemmModel,ceildiv
from .backends import make_backend
from .scheduler import operand_pipeline

def tile_for(shape,library):
    # Coarse software prior over our development observations; no timings in
    # the keys and no vendor table. This is not the vendor's exact selector.
    candidates=[x for x in library if x['supported'] and x.get('split_k',1)==1 and x['tile'][0]>=shape[0]]
    row=min(candidates,key=lambda x:(sum((a-b)**2 for a,b in zip(shape,x['shape'])),x['id']))
    return row['tile'],row['id']

def evaluate(shape,tile,hardware,policy,read_Bps,shared_read_Bpc,schedule=None):
    m,n,k=shape;bm,bn,bk=tile
    if not (1<=m<=32 and n%64==0 and k%64==0):raise ValueError('Decode domain: 1<=M<=32, N/K multiples of 64')
    h=hardware;p=policy;backend=make_backend('auto',h,p.matrix_efficiency);schedule=schedule or {}
    # Full two-slot operand pipeline, C remains in partitioned registers.
    mapping=Mapping(bm,bn,bk,1,1,2,2,4096,2,2)
    cap=GemmModel.capacity(mapping,h)
    if schedule.get('engine_partition'):
        pm,pn=schedule['engine_partition']
        if pm*pn!=h.arrays_per_core:raise ValueError('Engine partition does not match hardware')
        mi=ceildiv(bm,h.instruction_m*pm)*h.instruction_m
        ni=ceildiv(bn,h.instruction_n*pn)*h.instruction_n
        part_bytes=mi*ni*4+4*bk*(mi+ni)+ceildiv(h.register_reserve_bytes_per_core,h.arrays_per_core)
        cap['register_bytes']=part_bytes*h.arrays_per_core
        cap['operand_register_bytes']=4*bk*(mi+ni)*h.arrays_per_core
        cap['register_bytes_per_partition']=[part_bytes]*h.arrays_per_core
        if cap['register_bytes']>h.register_budget_bytes_per_core or (h.register_partitions_per_core and part_bytes>h.register_bytes_per_partition):
            cap['valid']=False;cap['rejections'].append('register-partition')
    if not cap['valid']:return None
    by_shared=h.shared_budget_bytes_per_core//cap['shared_bytes']
    by_register=h.register_budget_bytes_per_core//cap['register_bytes']
    if h.register_partitions_per_core:
        by_register=min(h.register_bytes_per_partition//v for v in cap['register_bytes_per_partition'])
    resident=max(1,min(by_shared,by_register))
    if schedule.get('task_waves'):
        resident=1
    # Per-task LDS and register allocations persist across K. Every CU may have
    # several such tasks in flight; their matrix/LDS services still share one CU.
    tasks=[(min(bm,m-i),min(bn,n-j)) for i in range(0,m,bm) for j in range(0,n,bn)]
    total=p.device_service_s+h.kernel_launch_s;details=[]
    count=ceildiv(k,bk)
    compute=backend.estimate(bm,bn,bk)
    operand_bytes=compute.shared_read_bytes
    if schedule.get('operand_reuse')=='fragment':
        fragments=ceildiv(bm,h.instruction_m)*ceildiv(bn,h.instruction_n)*ceildiv(bk,h.instruction_k)
        operand_bytes=fragments*(h.instruction_m+h.instruction_n)*h.instruction_k*2
    elif schedule.get('engine_partition'):
        pm,pn=schedule['engine_partition']
        if pm*pn!=h.arrays_per_core:raise ValueError('Engine partition does not match hardware')
        operand_bytes=2*bk*(pn*bm+pm*bn)
    for first in range(0,len(tasks),h.cores*resident):
        group=tasks[first:first+h.cores*resident];active=min(len(group),h.cores)
        maximum_tasks=ceildiv(len(group),h.cores)
        # Count the distinct inputs for this group. Decode presently has a
        # single M stripe for library templates; general stripe accounting is
        # used in the fallback mapper separately.
        if bm<m:raise ValueError('Resident Decode template requires bm>=M')
        width=sum(y for _,y in group)
        H=(m+width)*bk*2
        L=sum((x+y)*bk*2 for x,y in group)
        shared_store=max((x+y)*bk*2 for x,y in group)*maximum_tasks
        if schedule.get('padded_input_store'):shared_store=(bm+bn)*bk*2*maximum_tasks
        th=max(H/read_Bps,H/h.l2_write_bandwidth_Bps)
        tr=max(L/h.l2_read_bandwidth_Bps,shared_store/(h.shared_write_bytes_per_cycle_per_core*h.clock_hz))
        ts=maximum_tasks*operand_bytes/(shared_read_Bpc*h.clock_hz)
        tc=maximum_tasks*compute.duration_s
        # Group buffer counts represent separate equally sized slots in each
        # task. Shared service is accumulated per CU, never multiplied by CU count.
        pipe=operand_pipeline([(count,th,tr,ts,tc)],2,2,2,local_shared_port=schedule.get('shared_read_write_serialized',False))
        output_bytes=sum(x*y*2 for x,y in group)
        cu_output=max(x*y*2 for x,y in group)*maximum_tasks
        out=max(cu_output/(h.shared_write_bytes_per_cycle_per_core*h.clock_hz),
                output_bytes/h.l2_write_bandwidth_Bps,output_bytes/h.hbm_bandwidth_Bps)
        total+=pipe['duration_s']+out
        details.append(dict(task_count=len(group),active_cus=active,max_tasks_per_cu=maximum_tasks,
            input_hbm_bytes=H*count,l2_read_bytes=L*count,shared_input_write_bytes_per_cu=shared_store*count,
            shared_operand_bytes_per_cu=maximum_tasks*operand_bytes*count,
            service_us=dict(HBM=th*count*1e6,L2_Shared=tr*count*1e6,Shared_register=ts*count*1e6,MMAC=tc*count*1e6),
            duration_us=(pipe['duration_s']+out)*1e6,pipeline=pipe))
    return dict(shape=list(shape),tile=list(tile),duration_us=total*1e6,resident_tasks_per_cu=resident,
        capacity=cap,task_count=len(tasks),groups=details,mmac_issue_cycles=h.instruction_issue_cycles,
        input_read_Bps=read_Bps,shared_read_Bpc=shared_read_Bpc,schedule=schedule)

def predict(shape,profile=None):
    if profile is None:
        profile=json.loads((pathlib.Path(__file__).parent/'data/bw1100_decode_profile.json').read_text())
    if len(shape)!=3 or any(isinstance(x,bool) or not isinstance(x,int) or x<=0 for x in shape):
        raise ValueError('M, N and K must be positive integers')
    tile,source=tile_for(shape,profile['library'])
    h=Hardware(**profile['hardware']);p=Policy(**profile['policy'])
    schedule=profile.get('schedules',{}).get('x'.join(map(str,tile)),{})
    if profile.get('respect_prefetch_waves'):
        source_row=next(x for x in profile['library'] if x['id']==source)
        schedule={**schedule,'task_waves':source_row.get('prefetch_global',1)==1}
    result=evaluate(shape,tile,h,p,profile['input_read_Bps'],profile['shared_read_Bpc'],schedule)
    if result is None:raise ValueError('Nearest software template violates target resource capacity')
    result['dispatch_source']=source
    result['model_version']=profile['version']
    result['prediction_mode']='BW1100 measured-software-template prior + analytical service pipeline'
    return result
