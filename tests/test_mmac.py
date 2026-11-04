from dataclasses import replace
from pathlib import Path
import unittest
from superpod_compute import Hardware,GemmRequest,Mapping,evaluate_mapping
from superpod_compute.backends import MmacBackend,make_backend,ArrayBackend

class MmacTests(unittest.TestCase):
    def setUp(self):
        self.h=Hardware.load(Path(__file__).resolve().parents[1]/'superpod_compute/data/bw1100.json')
        self.b=MmacBackend(self.h)
    def test_peak_matches_independent_instruction_arithmetic(self):
        self.assertEqual(self.h.core_peak_flops*self.h.cores,383.3856e12)
        self.assertEqual(self.h.instruction_issue_cycles,8)
        self.assertIsInstance(make_backend('auto',self.h),MmacBackend)
        self.assertIsInstance(make_backend('auto',Hardware()),ArrayBackend)
    def test_single_fragment_cannot_use_four_engines(self):
        x=self.b.estimate(16,16,16)
        self.assertEqual(x.cycles,8);self.assertEqual(x.instruction_count,1);self.assertEqual(x.engines_used,1)
        self.assertEqual(self.b.estimate(16,16,160).cycles,80)
    def test_fragment_distribution_and_padding(self):
        self.assertEqual(self.b.estimate(32,32,16).cycles,8)
        self.assertEqual(self.b.estimate(48,32,16).cycles,16)
        self.assertEqual(self.b.estimate(17,19,17).cycles,16)
        self.assertEqual(self.b.estimate(256,384,16).cycles,768)
    def test_rectangular_engine_imbalance_is_not_hidden(self):
        # 9 by 7 fragments: best rectangular split is 1 by 4 engines,
        # with 9 by 2 fragments on its busiest engine; 17 K instructions.
        x=self.b.estimate(130,97,270)
        self.assertEqual(x.cycles,18*17*8)
        self.assertEqual((x.array_partition_m,x.array_partition_n),(1,4))
    def test_scales_with_instruction_service_not_mac_coefficient(self):
        original=self.b.estimate(128,128,128)
        # The physical-cell placeholder is not the MMAC backend's throughput knob.
        altered=MmacBackend(replace(self.h,mac_per_cell_per_cycle=9)).estimate(128,128,128)
        self.assertEqual(original,altered)
        slower=MmacBackend(replace(self.h,instruction_issue_cycles=16)).estimate(128,128,128)
        self.assertEqual(slower.cycles,2*original.cycles)
    def test_requires_explicit_profile(self):
        with self.assertRaises(ValueError):MmacBackend(Hardware())
        with self.assertRaises(ValueError):replace(self.h,instruction_issue_cycles=0)
        with self.assertRaises(ValueError):replace(self.h,matrix_k_quantum=32)

if __name__=='__main__':unittest.main()
