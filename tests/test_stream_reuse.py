from dataclasses import replace
import math, random, unittest
from superpod_compute import GemmRequest, Hardware, Policy, Mapping, evaluate_mapping
from superpod_compute.operators import GemmModel
from superpod_compute.scheduler import pipeline_segments, pipeline_events, EventScheduler
from superpod_compute.contracts import digest

class StreamTests(unittest.TestCase):
    def test_calibration_cannot_silently_move_to_other_hardware(self):
        h=Hardware();policy=Policy(calibration_hardware_sha256=digest(h.to_dict()),device_service_s=3e-6)
        q=GemmRequest(16,16,16);p=Mapping(16,16,16,1,1,1,1)
        evaluate_mapping(q,p,h,policy=policy)
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            evaluate_mapping(q,p,replace(h,cores=72),policy=policy)

    def test_piecewise_pipeline_against_explicit_dag(self):
        rng=random.Random(25)
        for _ in range(30):
            segments=[(rng.randrange(1,30),*(rng.random()*1e-6 for _ in range(3))) for _ in range(3)]
            segments.insert(1,(7,0,2e-6,1e-6))
            triples=[(h,r,c) for count,h,r,c in segments for _ in range(count)]
            for L in (1,2):
                for S in (1,2):
                    for shared in (True,False):
                        result=pipeline_segments(segments,L,S,shared_port=shared)
                        ref=EventScheduler().run(pipeline_events(triples,L,S,shared))
                        self.assertTrue(math.isclose(result['duration_s'],ref[-1]['end_s'],rel_tol=1e-11,abs_tol=1e-15))

    def test_large_output_fits_streaming_but_registers_still_required(self):
        h=Hardware.load('superpod_compute/data/bw1100.json');p=Mapping(256,384,16,8,9,2,2)
        cap=GemmModel.capacity(p,h)
        self.assertTrue(cap['valid']);self.assertEqual(cap['shared_input_bytes'],40*1024)
        self.assertEqual(cap['shared_output_bytes'],32*1024)
        self.assertGreater(p.bm*p.bn*2,h.shared_budget_bytes_per_core)
        self.assertIn('registers',GemmModel.capacity(p,replace(h,register_budget_bytes_per_core=256*1024))['rejections'])

    def test_chunks_cover_output_and_hold_slots_until_drain(self):
        h=Hardware.load('superpod_compute/data/bw1100.json');p=Mapping(256,384,16,2,2,2,2)
        r=evaluate_mapping(GemmRequest(400,600,32),p,h,trace_limit=100)
        for w in r['wave_classes']:
            o=w['output_stream'];self.assertEqual(sum(x['chunks']*x['bytes_per_chunk_all_cores'] for x in o['byte_runs']),w['valid_m']*w['valid_n']*2)
            trace=o['pipeline']['trace']
            for i,row in enumerate(trace):
                if i>=2:
                    self.assertGreaterEqual(row['W_start_s'],trace[i-2]['L_end_s'])
                    self.assertGreaterEqual(row['L_start_s'],trace[i-2]['D_end_s'])
            self.assertLessEqual(o['duration_s'],o['register_to_shared_service_s']+w['write_to_l2_s']+w['write_to_hbm_s']+1e-15)
        serial=evaluate_mapping(GemmRequest(400,600,32),replace(p,output_buffers=1),h)
        self.assertGreaterEqual(serial['duration_s'],r['duration_s'])

    def test_reuse_known_traffic_with_no_l2_to_core_discount(self):
        q=GemmRequest(3072,2304,1024);p=Mapping(128,128,128,12,9,2,2)
        cold=evaluate_mapping(q,p,policy=Policy(input_reuse=False))
        hit=evaluate_mapping(q,p)
        self.assertEqual(cold['hbm_read_bytes'],21*1024**2)
        self.assertEqual(hit['hbm_read_bytes'],15*1024**2)
        self.assertEqual(hit['l2_to_core_bytes'],cold['l2_to_core_bytes'])
        self.assertEqual(hit['hbm_bytes_saved_by_reuse'],6*1024**2)

    def test_partial_retention_releases_capacity_and_changes_panel_bytes(self):
        h=replace(Hardware(),l2_bytes=40000,cores=1)
        p=Mapping(32,32,16,1,1,2,2)
        q=GemmRequest(32,64,1024)
        result=evaluate_mapping(q,p,h)
        self.assertEqual(result['reuse_order'],'A')
        hit=[w for w in result['wave_classes'] if w['is_reuse_wave']][0]
        self.assertGreater(hit['retained_k'],0);self.assertLess(hit['retained_k'],1024)
        self.assertLessEqual(hit['retained_k']*32*2,result['retention_budget_bytes'])
        runs=hit['panel_byte_runs']
        self.assertEqual(runs[0]['HBM_bytes_per_panel'],32*16*2)
        self.assertEqual(runs[1]['HBM_bytes_per_panel'],(32+32)*16*2)
        self.assertEqual(sum(x['panels']*x['k_elements'] for x in runs),1024)

    def test_warm_state_and_device_cost_are_explicit(self):
        q=GemmRequest(128,128,128);p=Mapping(32,32,32,4,4,2,2)
        base=evaluate_mapping(q,p)
        warm=evaluate_mapping(q,p,policy=Policy(initial_cache_state='warm-if-fit',device_service_s=3e-6))
        self.assertEqual(warm['hbm_read_bytes'],0)
        self.assertEqual(warm['l2_to_core_bytes'],base['l2_to_core_bytes'])
        without=evaluate_mapping(q,p,policy=Policy(initial_cache_state='warm-if-fit'))
        self.assertAlmostEqual(warm['duration_s']-without['duration_s'],3e-6)
        small_l2=evaluate_mapping(q,p,replace(Hardware(),l2_bytes=80000),policy=Policy(initial_cache_state='warm-if-fit'))
        self.assertFalse(small_l2['warm_residency_admitted']);self.assertGreater(small_l2['hbm_read_bytes'],0)

if __name__=='__main__':unittest.main()
