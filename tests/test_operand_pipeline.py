from dataclasses import replace
import math,random,unittest
from superpod_compute import Hardware,Mapping
from superpod_compute.operators import GemmModel
from superpod_compute.scheduler import Event,EventScheduler,operand_pipeline

class OperandTests(unittest.TestCase):
    def test_against_independent_four_stage_event_dag(self):
        rng=random.Random(86)
        for _ in range(12):
            segments=[(rng.randint(1,20),*(rng.random()*1e-6 for _ in range(4))) for _ in range(3)]
            rows=[(h,r,s,c) for num,h,r,s,c in segments for _ in range(num)]
            for L in (1,2):
                for S in (1,2):
                    for V in (1,2):
                        for shared in (False,True):
                            ev=[]
                            for i,(h,r,s,c) in enumerate(rows):
                                for j,(name,dt) in enumerate(zip('HRSC',(h,r,s,c))):
                                    deps=[f'{"HRSC"[j-1]}{i}'] if j else []
                                    if j<3 and i>=(L,S,V)[j]:deps.append(f'{"HRSC"[j+1]}{i-(L,S,V)[j]}')
                                    resource='HR' if shared and j<2 else name
                                    ev.append(Event(f'{name}{i}',dt,(resource,),tuple(deps)))
                            ref=EventScheduler().run(ev)[-1]['end_s']
                            actual=operand_pipeline(segments,L,S,V,shared_port=shared)['duration_s']
                            self.assertTrue(math.isclose(ref,actual,rel_tol=1e-11,abs_tol=1e-15))
    def test_register_buffer_lifetime_and_accumulator_capacity(self):
        run=operand_pipeline([(12,1e-6,1e-6,2e-6,3e-6)],2,2,2,trace_limit=12)
        t=run['trace']
        for i in range(2,len(t)):
            self.assertGreaterEqual(t[i]['R_start_s'],t[i-2]['S_end_s'])
            self.assertGreaterEqual(t[i]['S_start_s'],t[i-2]['C_end_s'])
        h=replace(Hardware(),matrix_backend='mmac',shared_read_bytes_per_cycle_per_core=64,register_budget_bytes_per_core=512*1024)
        p=Mapping(256,384,16,1,1,2,2)
        cap=GemmModel.capacity(p,h)
        self.assertEqual(cap['operand_register_bytes'],80*1024)
        self.assertEqual(cap['register_bytes'],496*1024)
    def test_large_k_exact_acceleration(self):
        x=operand_pipeline([(10**8,1e-6,2e-6,3e-6,4e-6)],2,2,2)
        self.assertGreater(x['skipped_panels'],10**8-30)
        self.assertAlmostEqual(x['duration_s'],6e-6+4e-6*10**8,places=6)

if __name__=='__main__':unittest.main()
