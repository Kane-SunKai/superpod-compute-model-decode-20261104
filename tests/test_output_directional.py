import math,random,unittest
from dataclasses import replace
from superpod_compute import Hardware,GemmRequest,Mapping,Policy,evaluate_mapping
from superpod_compute.backends import MmacBackend,MacThroughputBackend
from superpod_compute.scheduler import output_pipeline,Event,EventScheduler

class DirectionalOutputTests(unittest.TestCase):
    def test_output_scheduler_against_independent_event_dag(self):
        rng=random.Random(29)
        for _ in range(12):
            segments=[(rng.randint(1,17),*(rng.random()*1e-6 for _ in range(3))) for j in range(3)]
            for S in (1,2):
                for L in (1,2):
                    for shared in (False,True):
                        events=[];i=0
                        for num,w,l,d in segments:
                            for j in range(num):
                                events.extend([
                                  Event(f'W{i}',w,('write-shared',),(f'L{i-S}',) if i>=S else ()),
                                  Event(f'L{i}',l,('port' if shared else 'l2',),(f'W{i}',)+((f'D{i-L}',) if i>=L else ())),
                                  Event(f'D{i}',d,('port' if shared else 'hbm',),(f'L{i}',))])
                                i+=1
                        ref=EventScheduler().run(events)[-1]['end_s']
                        run=output_pipeline(segments,S,L,shared_port=shared,trace_limit=100)
                        self.assertTrue(math.isclose(ref,run['duration_s'],rel_tol=1e-11))
                        for j,t in enumerate(run['trace']):
                            if j>=S:self.assertGreaterEqual(t['W_start_s'],run['trace'][j-S]['L_end_s'])
                            if j>=L:self.assertGreaterEqual(t['L_start_s'],run['trace'][j-L]['D_end_s'])
    def test_accelerated_output_and_tail(self):
        segments=[(1000,1e-6,2e-6,3e-6),(1,.5e-6,1e-6,1.5e-6)]
        fast=output_pipeline(segments,2,2)
        slow=output_pipeline(segments,2,2,accelerate=False)
        self.assertAlmostEqual(fast['duration_s'],slow['duration_s'],places=13)
        self.assertGreater(fast['skipped_panels'],900)
    def test_directional_fields_fallback_and_validation(self):
        h=Hardware()
        self.assertEqual(h.l2_read_bandwidth_Bps,h.l2_bandwidth_Bps)
        self.assertEqual(h.l2_write_bandwidth_Bps,h.l2_bandwidth_Bps)
        x=replace(h,l2_read_bytes_per_cycle=3768,l2_write_bytes_per_cycle=2990)
        self.assertEqual(x.l2_read_bandwidth_Bps,3768*x.clock_hz)
        self.assertEqual(x.l2_write_bandwidth_Bps,2990*x.clock_hz)
        with self.assertRaises(ValueError):replace(h,shared_write_bytes_per_cycle_per_core=-1)
    def test_mac_comparison_preserves_dataflow_and_only_changes_compute(self):
        h=Hardware.load('superpod_compute/data/bw1100.json')
        mac=replace(h,matrix_backend='mac-throughput',mac_per_cell_per_cycle=1)
        b=MmacBackend(h);a=MacThroughputBackend(mac)
        for shape in [(256,384,16),(130,97,270),(16,16,16),(1,8192,16)]:
            x,y=b.estimate(*shape),a.estimate(*shape)
            self.assertEqual(y.cycles,2*x.cycles)
            self.assertEqual(x.shared_read_bytes,y.shared_read_bytes)
            self.assertEqual(x.engines_used,y.engines_used)
        self.assertEqual(mac.core_peak_flops*mac.cores,191.6928e12)
    def test_write_service_local_not_multiplied_by_cu_count(self):
        h=Hardware.load('superpod_compute/data/bw1100.json')
        p=Mapping(256,384,16,12,6,1,2,4096,2,2)
        q=GemmRequest(3072,2304,1024)
        r=evaluate_mapping(q,p,h,trace_limit=64)
        w=r['wave_classes'][0];o=w['output_stream'];b=o['byte_runs'][0]
        self.assertAlmostEqual(b['register_to_shared_s'],4096/(h.shared_write_bytes_per_cycle_per_core*h.clock_hz))
        self.assertAlmostEqual(b['l2_write_s'],72*4096/h.l2_write_bandwidth_Bps)
        self.assertEqual(o['pipeline']['panels'],48)
        self.assertEqual(sum(x['chunks']*x['bytes_per_chunk_all_cores'] for x in o['byte_runs']),3072*2304*2)
        self.assertFalse(o['output_shared_read_charged'])
        off=evaluate_mapping(q,p,replace(h,shared_write_bytes_per_cycle_per_core=0))
        self.assertGreater(r['duration_s'],off['duration_s'])

if __name__=='__main__':unittest.main()
