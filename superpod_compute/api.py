"""Generic candidate search; GEMM is the first registered operator plugin."""
from collections import Counter
import math
from .backends import make_backend
from .cache import MappingCache
from .contracts import Hardware, Policy, Mapping, MODEL_VERSION, SEARCH_VERSION, digest
from .operators import default_registry

def signature_for(request,hardware,policy,backend,model):
    return {'request':request.to_dict(),'hardware':hardware.to_dict(),'policy':policy.to_dict(),'backend':backend.signature,'operator_model':model.version,'model_version':MODEL_VERSION,'search_version':SEARCH_VERSION,'initial_state':policy.initial_cache_state,'dataflow':'resident-C-full-K; streamed-output; capacity-bounded-input-reuse; sequential-waves'}

def predict(request,hardware=None,*,policy=None,backend='auto',cache_path=None,force=False,registry=None,progress=None,lut_path=None):
    hardware=hardware or Hardware.load();policy=policy or Policy();registry=registry or default_registry();model=registry.get(request.operator);model.validate(request,hardware)
    if policy.calibration_hardware_sha256 and policy.calibration_hardware_sha256!=digest(hardware.to_dict()):raise ValueError('Calibration hardware fingerprint mismatch; recalibrate device service')
    compute=make_backend(backend,hardware,policy.matrix_efficiency,lut_path) if isinstance(backend,str) else backend
    signature=signature_for(request,hardware,policy,compute,model);key=digest(signature)
    special=model.special(request,hardware,policy)
    if special is not None:return {'status':special['status'],'signature':signature,'signature_sha256':key,'cache_hit':False,'special':special,'best':None,'searched_candidates':0,'feasible_candidates':0}
    cache=MappingCache(cache_path) if cache_path else None
    if cache and not force:
        saved=cache.get(signature)
        if saved:
            try:
                mapping=model.mapping_from_dict(saved['best']['mapping'])
                valid=any(candidate==mapping for candidate in model.candidates(request,hardware,policy))
                checked=model.evaluate(request,hardware,policy,compute,mapping) if valid else None
                if checked and math.isclose(checked['duration_s'],saved['best']['duration_s'],rel_tol=1e-12,abs_tol=1e-15):
                    return {**saved,'cache_hit':True}
            except (KeyError,TypeError,ValueError):pass
    top=[];count=feasible=0;rejected=Counter()
    for mapping in model.candidates(request,hardware,policy):
        count+=1
        cap=model.capacity(mapping,hardware)
        if not cap['valid']:
            rejected.update(cap['rejections'])
        else:
            evaluated=model.evaluate(request,hardware,policy,compute,mapping);feasible+=1
            if evaluated is not None:
                if not math.isfinite(evaluated['duration_s']):raise ValueError('Nonfinite candidate estimate')
                top.append(evaluated);top.sort(key=model.rank_key)
                if len(top)>policy.top_k:top.pop()
        if progress and count%256==0:progress(count,feasible)
    if not top:raise ValueError(f'No feasible mapping in candidate space; rejected constraints: {dict(rejected)}')
    result={'status':'predicted','signature':signature,'signature_sha256':key,'cache_hit':False,'best':top[0],'top_candidates':top,'searched_candidates':count,'feasible_candidates':feasible,'rejected_constraint_counts':dict(rejected),'optimality':'minimum of the enumerated feasible candidates under this cost model','validation_status':'model-logic tested; not calibrated against physical A100'}
    if cache:cache.put(signature,result)
    return result

def evaluate_mapping(request,mapping,hardware=None,*,policy=None,backend='auto',trace_limit=8):
    hardware=hardware or Hardware.load();policy=policy or Policy();model=default_registry().get(request.operator);model.validate(request,hardware)
    if policy.calibration_hardware_sha256 and policy.calibration_hardware_sha256!=digest(hardware.to_dict()):raise ValueError('Calibration hardware fingerprint mismatch; recalibrate device service')
    compute=make_backend(backend,hardware,policy.matrix_efficiency) if isinstance(backend,str) else backend
    result=model.evaluate(request,hardware,policy,compute,mapping,trace_limit=trace_limit)
    if result is None:raise ValueError('Mapping violates resource budgets: '+str(model.capacity(mapping,hardware)))
    return result
