"""Run with python -m examples.demo from the repository root."""
from superpod_compute import GemmRequest, Mapping, predict, evaluate_mapping

request=GemmRequest(3072,2304,1024)
result=predict(request,cache_path='.cache/mappings.sqlite3')
print('cache_hit:',result['cache_hit'])
print('best:',result['best']['mapping'])
print('model duration (us):',result['best']['duration_us'])
fixed=evaluate_mapping(request,Mapping(128,128,128,12,9,2,2),trace_limit=8)
print('fixed teaching mapping duration (us):',fixed['duration_us'])
print('first wave pipeline:',fixed['wave_classes'][0]['pipeline'])
