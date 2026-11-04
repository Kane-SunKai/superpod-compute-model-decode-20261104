import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from contextlib import closing

from superpod_compute import GemmRequest, Hardware, Policy, Mapping, predict, evaluate_mapping
from superpod_compute.api import signature_for
from superpod_compute.backends import ArrayBackend, PeakBackend
from superpod_compute.cache import MappingCache
from superpod_compute.contracts import digest
from superpod_compute.operators import GemmModel, default_registry, OperatorRegistry
from superpod_compute.scheduler import Event, EventScheduler, pipeline, pipeline_events

SMALL=Policy(bm_values=(16,32),bn_values=(16,32),bk_values=(16,32),add_shape_candidate=False)

class ContractsTests(unittest.TestCase):
    def test_hardware_defaults_and_profile(self):
        h=Hardware.load();self.assertEqual(h,Hardware());self.assertAlmostEqual(h.core_peak_flops*108,311.86944e12);self.assertEqual(h.l2_bandwidth_Bps,7.2192e12)
    def test_invalid_inputs(self):
        for args in [(-1,3,4),(True,3,4),(1.5,3,4)]:
            with self.assertRaises(ValueError):GemmRequest(*args)
        with self.assertRaises(ValueError):GemmRequest(1,2,3,dtype='fp32')
        for kwargs in [{'clock_hz':0},{'hbm_bandwidth_Bps':math.nan},{'cores':1.5},{'shared_budget_bytes_per_core':300*1024}]:
            with self.assertRaises(ValueError):Hardware(**kwargs)
        for kwargs in [{'matrix_efficiency':2},{'buffer_counts':(3,)},{'bm_values':()},{'charge_hbm_latency':'yes'}]:
            with self.assertRaises(ValueError):Policy(**kwargs)
    def test_hbm_capacity(self):
        with self.assertRaisesRegex(ValueError,'HBM'):predict(GemmRequest(10**9,10**9,10**9),policy=SMALL)
    def test_degenerate_shapes(self):
        self.assertEqual(predict(GemmRequest(0,100,100))['special']['duration_s'],0)
        z=predict(GemmRequest(7,13,0));self.assertEqual(z['status'],'zero-fill');self.assertEqual(z['special']['output_bytes'],7*13*2)
    def test_no_feasible_mapping(self):
        with self.assertRaisesRegex(ValueError,'No feasible'):predict(GemmRequest(16,16,16),replace(Hardware(),register_budget_bytes_per_core=32*1024),policy=SMALL)
    def test_operator_is_not_silently_supported(self):
        with self.assertRaises(NotImplementedError):default_registry().get('softmax')
    def test_generic_plugin_has_no_gemm_mapping_dependency(self):
        class Request:
            operator='test_vector'
            def to_dict(self):return {'operator':self.operator,'length':32}
        class Backend:
            signature={'backend':'test-vector-service'}
        class Plugin:
            name='test_vector';version='test-v1'
            def validate(self,*args):pass
            def special(self,*args):return None
            def candidates(self,*args):return iter((1,2,4))
            def capacity(self,*args):return {'valid':True,'rejections':[]}
            def mapping_from_dict(self,value):return value['lanes']
            def rank_key(self,value):return value['duration_s']
            def evaluate(self,request,hardware,policy,backend,mapping,trace_limit=0):return {'mapping':{'lanes':mapping},'duration_s':32/mapping,'capacity':{'valid':True}}
        registry=OperatorRegistry();registry.register(Plugin())
        with tempfile.TemporaryDirectory() as d:
            kwargs={'registry':registry,'backend':Backend(),'cache_path':Path(d)/'custom.sqlite3'}
            a=predict(Request(),**kwargs);b=predict(Request(),**kwargs)
            self.assertEqual(a['best']['mapping'],{'lanes':4});self.assertTrue(b['cache_hit'])

class ArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.backend=ArrayBackend(Hardware())
    def test_known_core_example(self):
        e=self.backend.estimate(128,128,128);self.assertEqual(e.cycles,2069);self.assertEqual(e.source,'llmcompass-fast-0.99');self.assertEqual(e.array_partition_m*e.array_partition_n,4)
    def test_known_lut_example(self):
        cycles,source=self.backend.array_cycles(16,16,32);self.assertEqual(cycles,61);self.assertEqual(source,'llmcompass-lut')
    def test_os_model_against_every_reference_entry(self):
        self.assertEqual(len(self.backend.table),1049)
        for (m,n,k,h,w,flow),cycles in self.backend.table.items():
            self.assertEqual(flow,'os');self.assertEqual(ArrayBackend.os_cycles(m,n,k,h,w),cycles)
    def test_missing_shape_is_labeled(self):
        cycles,source=self.backend.array_cycles(17,19,13);self.assertGreater(cycles,0);self.assertEqual(source,'own-os-wavefront-estimate')
    def test_peak_reference(self):
        self.assertEqual(PeakBackend(Hardware()).estimate(128,128,128).cycles,2048)
    def test_invalid_tile(self):
        with self.assertRaises(ValueError):self.backend.estimate(0,16,16)
    def test_clock_and_efficiency(self):
        a=self.backend.estimate(128,128,128);slow=ArrayBackend(replace(Hardware(),clock_hz=Hardware().clock_hz/2)).estimate(128,128,128)
        self.assertEqual(a.cycles,slow.cycles);self.assertAlmostEqual(slow.duration_s,a.duration_s*2)
        self.assertGreaterEqual(ArrayBackend(Hardware(),.5).estimate(128,128,128).cycles,2*a.cycles)
    def test_rectangular_hardware(self):
        b=ArrayBackend(replace(Hardware(),array_rows=8,array_cols=32));self.assertGreater(b.estimate(16,32,16).cycles,0)

class SchedulingTests(unittest.TestCase):
    def test_fast_pipeline_against_generic_dag(self):
        rng=random.Random(41)
        for _ in range(60):
            h,r,c=[rng.random()*5e-6 for _ in range(3)];count=rng.randint(1,80);tail=tuple(rng.random()*x for x in (h,r,c))
            for L in (1,2):
                for S in (1,2):
                    for shared in (False,True):
                        fast=pipeline(h,r,c,count,L,S,tail,shared_port=shared)
                        events=pipeline_events([(h,r,c)]*count+[tail],L,S,shared)
                        reference=EventScheduler().run(events)
                        self.assertTrue(math.isclose(fast['duration_s'],reference[-1]['end_s'],rel_tol=1e-11,abs_tol=1e-15))
    def test_slot_lifetimes_and_resources(self):
        for L in (1,2):
            for S in (1,2):
                t=pipeline(3e-6,1e-6,4e-6,20,L,S,trace_limit=20)['trace']
                for i,e in enumerate(t):
                    self.assertGreaterEqual(e['R_start_s'],e['H_end_s']);self.assertGreaterEqual(e['C_start_s'],e['R_end_s'])
                    if i>=L:self.assertGreaterEqual(e['H_start_s'],t[i-L]['R_end_s'])
                    if i>=S:self.assertGreaterEqual(e['R_start_s'],t[i-S]['C_end_s'])
                    if i:
                        for stage in ('H','R','C'):self.assertGreaterEqual(e[stage+'_start_s'],t[i-1][stage+'_end_s'])
    def test_large_k_fast_forward(self):
        x=pipeline(1e-6,2e-6,3e-6,10**9,2,2,trace_limit=8);self.assertGreater(x['skipped_panels'],10**9-20);self.assertEqual(len(x['trace']),8);self.assertAlmostEqual(x['duration_s'],1e-6+2e-6+3e-6*10**9,places=6)
    def test_single_panel_buffers(self):
        results=[pipeline(1,2,3,1,L,S)['duration_s'] for L in (1,2) for S in (1,2)];self.assertEqual(results,[6]*4)
    def test_shared_port_removes_transfer_overlap(self):
        a=pipeline(3,2,1,10,2,2);b=pipeline(3,2,1,10,2,2,shared_port=True);self.assertGreater(b['duration_s'],a['duration_s'])
    def test_general_resources_and_dependencies(self):
        s=EventScheduler().run([Event('load',2,('memory',)),Event('reduce',3,('vector',),('load',)),Event('next_load',2,('memory',)),Event('normalize',1,('vector',),('reduce',))]);self.assertEqual(s[2]['start_s'],2);self.assertEqual(s[3]['end_s'],6)
        with self.assertRaises(ValueError):EventScheduler().run([Event('bad',1,(),('missing',))])

class GemmTests(unittest.TestCase):
    def test_capacity_rules(self):
        model=GemmModel();h=Hardware();a=model.capacity(Mapping(128,128,128,12,9,2,2),h)
        self.assertEqual(a['shared_bytes'],128*1024);self.assertEqual(a['register_bytes'],96*1024);self.assertTrue(a['valid'])
        self.assertFalse(model.capacity(Mapping(128,128,256,12,9,2,2),h)['valid'])
        self.assertTrue(model.capacity(Mapping(128,128,256,12,9,2,1),h)['valid'])
    def test_known_traffic(self):
        r=evaluate_mapping(GemmRequest(3072,2304,1024),Mapping(128,128,128,12,9,2,2),policy=Policy(input_reuse=False));self.assertEqual(r['hbm_read_bytes'],21*1024**2);self.assertEqual(r['l2_to_core_bytes'],216*1024**2);self.assertEqual(r['output_bytes'],13.5*1024**2);self.assertEqual(r['wave_count'],4)
    def test_irregular_wave_aggregation(self):
        req=GemmRequest(130,97,270);p=Mapping(64,64,128,2,1,2,2);h=Hardware();combined=evaluate_mapping(req,p,h,policy=Policy(input_reuse=False));time=h.kernel_launch_s;hb=rb=0
        for m in range(0,req.M,128):
            for n in range(0,req.N,64):
                r=evaluate_mapping(GemmRequest(min(128,req.M-m),min(64,req.N-n),req.K),p,h,policy=Policy(input_reuse=False));time+=r['duration_s']-h.kernel_launch_s;hb+=r['hbm_read_bytes'];rb+=r['l2_to_core_bytes']
        self.assertAlmostEqual(time,combined['duration_s']);self.assertEqual(hb,combined['hbm_read_bytes']);self.assertEqual(rb,combined['l2_to_core_bytes']);self.assertGreater(combined['executed_tile_flops'],combined['useful_flops'])
    def test_tail_valid_bytes_and_pad_compute(self):
        r=evaluate_mapping(GemmRequest(2,3,14),Mapping(16,16,128,1,1,1,1));self.assertEqual(r['hbm_read_bytes'],(2+3)*14*2);self.assertEqual(r['output_bytes'],12);self.assertEqual(r['executed_tile_flops'],2*16*16*16)
    def test_tiling_math_coverage(self):
        for M,N,K in [(1,7,3),(17,19,33),(31,23,15),(35,34,39)]:
            A=[[((i+k)%5)-2 for k in range(K)] for i in range(M)];B=[[((k*3+j)%7)-3 for j in range(N)] for k in range(K)];expected=[[sum(A[i][k]*B[k][j] for k in range(K)) for j in range(N)] for i in range(M)];out=[[0]*N for _ in range(M)]
            for m in range(0,M,16):
                for n in range(0,N,16):
                    for k0 in range(0,K,16):
                        for i in range(m,min(m+16,M)):
                            for j in range(n,min(n+16,N)):
                                for k in range(k0,min(k0+16,K)):out[i][j]+=A[i][k]*B[k][j]
            self.assertEqual(out,expected)
    def test_exhaustive_minimum(self):
        req=GemmRequest(130,97,270);h=Hardware();result=predict(req,h,policy=SMALL);backend=ArrayBackend(h);model=GemmModel();scores=[r for p in model.candidates(req,h,SMALL) if (r:=model.evaluate(req,h,SMALL,backend,p)) is not None]
        self.assertEqual(result['best'],min(scores,key=model.rank_key));self.assertEqual(result['feasible_candidates'],len(scores))
    def test_one_panel_prefers_single_buffers(self):
        r=predict(GemmRequest(128,128,16),policy=SMALL);self.assertEqual(r['best']['mapping']['l2_buffers'],1);self.assertEqual(r['best']['mapping']['shared_buffers'],1)
    def test_low_parallelism_not_full_chip_peak(self):
        h=Hardware();p=Mapping(16,16,16,1,1,1,1);a=evaluate_mapping(GemmRequest(16,16,16),p,h,backend='peak');b=evaluate_mapping(GemmRequest(16,16,16),p,replace(h,cores=1),backend='peak');self.assertEqual(a['duration_s'],b['duration_s'])

class CacheTests(unittest.TestCase):
    def test_hit_and_invalidations(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d)/'m.sqlite3';req=GemmRequest(37,61,33);a=predict(req,policy=SMALL,cache_path=cache);b=predict(req,policy=SMALL,cache_path=cache);self.assertFalse(a['cache_hit']);self.assertTrue(b['cache_hit']);self.assertEqual(a['best'],b['best'])
            changes=[(GemmRequest(38,61,33),Hardware(),SMALL,'array'),(req,replace(Hardware(),clock_hz=1e9),SMALL,'array'),(req,Hardware(),SMALL,'peak'),(req,Hardware(),replace(SMALL,transfer_resource_mode='shared-port'),'array'),(replace(req,dtype='bf16'),Hardware(),SMALL,'array')]
            for q,h,p,backend in changes:self.assertFalse(predict(q,h,policy=p,backend=backend,cache_path=cache)['cache_hit'])
            self.assertEqual(MappingCache(cache).count(),6)
    def test_corrupt_payload_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.sqlite3';r=predict(GemmRequest(16,16,16),policy=SMALL,cache_path=path)
            with closing(sqlite3.connect(path)) as db:
                db.execute("UPDATE mappings SET result='{}'");db.commit()
            self.assertFalse(predict(GemmRequest(16,16,16),policy=SMALL,cache_path=path)['cache_hit'])
    def test_force_and_clear(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.sqlite3';q=GemmRequest(16,16,16);predict(q,policy=SMALL,cache_path=path);self.assertFalse(predict(q,policy=SMALL,cache_path=path,force=True)['cache_hit']);c=MappingCache(path);c.clear();self.assertEqual(c.count(),0)
    def test_lut_fingerprint(self):
        req=GemmRequest(16,16,16);h=Hardware();a=ArrayBackend(h);first=signature_for(req,h,SMALL,a,GemmModel());a.lut_hash='different';second=signature_for(req,h,SMALL,a,GemmModel());self.assertNotEqual(digest(first),digest(second))

if __name__=='__main__':unittest.main()
