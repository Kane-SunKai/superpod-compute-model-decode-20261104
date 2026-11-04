"""Input calibration must not silently change output traffic or write service."""
from dataclasses import replace
import unittest
from superpod_compute.contracts import GemmRequest,Hardware,Policy,Mapping
from superpod_compute.backends import make_backend
from superpod_compute.operators import GemmModel

class HalfReadRate(GemmModel):
    def input_hbm_bandwidth(self,hardware):return hardware.hbm_bandwidth_Bps/2

class InputServiceTests(unittest.TestCase):
    def test_read_only_override_preserves_writes(self):
        h=replace(Hardware(),cores=4,hbm_bandwidth_Bps=1e9,kernel_launch_s=0)
        req=GemmRequest(128,128,256);p=Policy(initial_cache_state='cold')
        m=Mapping(64,64,32,2,2,2,2);be=make_backend('auto',h)
        original=GemmModel().evaluate(req,h,p,be,m)
        slower=HalfReadRate().evaluate(req,h,p,be,m)
        self.assertEqual(original['hbm_read_bytes'],slower['hbm_read_bytes'])
        self.assertEqual(original['output_bytes'],slower['output_bytes'])
        self.assertEqual(original['wave_classes'][0]['output_stream'],slower['wave_classes'][0]['output_stream'])
        self.assertGreater(slower['duration_s'],original['duration_s'])
    def test_hot_input_is_unaffected_by_read_bandwidth(self):
        h=replace(Hardware(),cores=4,kernel_launch_s=0)
        req=GemmRequest(128,128,256);p=Policy(initial_cache_state='warm-if-fit')
        m=Mapping(64,64,32,2,2,2,2);be=make_backend('auto',h)
        a=GemmModel().evaluate(req,h,p,be,m);b=HalfReadRate().evaluate(req,h,p,be,m)
        self.assertTrue(a['warm_residency_admitted'])
        self.assertEqual(a['duration_s'],b['duration_s'])
if __name__=='__main__':unittest.main()
