import unittest
from superpod_compute.scheduler import operand_pipeline

class SharedPortTests(unittest.TestCase):
    def test_shared_input_and_operand_services_serialize(self):
        # 100 panels, a two-unit store and three-unit read. Independent service
        # reaches 3 units/panel; one shared port reaches 5 units/panel.
        free=operand_pipeline([(100,0.,2.,3.,0.)],2,2,2)
        shared=operand_pipeline([(100,0.,2.,3.,0.)],2,2,2,local_shared_port=True)
        self.assertEqual(free['duration_s'],302.)
        self.assertEqual(shared['duration_s'],500.)
    def test_fast_path_equals_explicit_iteration(self):
        segments=[(123,1.,2.,3.,1.7),(37,.5,3.,1.,4.)]
        for shared in (True,False):
            a=operand_pipeline(segments,2,2,2,local_shared_port=shared)
            b=operand_pipeline(segments,2,2,2,local_shared_port=shared,accelerate=False)
            self.assertAlmostEqual(a['duration_s'],b['duration_s'],places=9)
