import dataclasses,json,pathlib,unittest
from superpod_compute.decode import evaluate,predict
from superpod_compute.contracts import Hardware,Policy

class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=json.loads((pathlib.Path(__file__).parents[1]/'superpod_compute/data/bw1100_decode_profile.json').read_text())
        cls.h=Hardware(**cls.profile['hardware']);cls.p=Policy(**cls.profile['policy'])
    def run_shape(self,n,waves):
        return evaluate([2,n,6656],[16,128,32],self.h,self.p,1.83e12,58,dict(task_waves=waves))
    def test_tail_wave_keeps_local_work(self):
        full=self.run_shape(9216,True);tail=self.run_shape(9472,True)
        self.assertEqual([x['task_count'] for x in tail['groups']],[72,2])
        self.assertGreater(tail['duration_us']-full['duration_us'],25)
        self.assertAlmostEqual(tail['groups'][1]['service_us']['Shared_register'],full['groups'][0]['service_us']['Shared_register'])
    def test_residency_is_resource_constrained(self):
        a=self.run_shape(9472,False)
        self.assertEqual(len(a['groups']),1)
        self.assertLessEqual(a['capacity']['shared_bytes']*a['resident_tasks_per_cu'],self.h.shared_budget_bytes_per_core)
        for used in a['capacity']['register_bytes_per_partition']:
            self.assertLessEqual(used*a['resident_tasks_per_cu'],192*1024)
    def test_invalid_domain(self):
        for shape in ([0,8192,8192],[1,8191,8192],[33,8192,8192],[True,8192,8192]):
            with self.assertRaises(ValueError):predict(shape,self.profile)
    def test_prediction_does_not_consume_timing_columns(self):
        before=predict([2,8192,8192],self.profile)
        other=json.loads(json.dumps(self.profile))
        for x in other['library']:x['measured_us']=999999
        after=predict([2,8192,8192],other)
        self.assertEqual(before['duration_us'],after['duration_us'])
    def test_hbm_bytes_conserved(self):
        a=self.run_shape(9472,True)
        # B read once; A is reused per wave, hence exactly two A reads.
        self.assertEqual(sum(x['input_hbm_bytes'] for x in a['groups']),2*6656*(9472+2*2))

if __name__=='__main__':unittest.main()
