import dataclasses,unittest
from superpod_compute.contracts import Hardware,Mapping
from superpod_compute.operators import GemmModel

class RegisterPartitionTests(unittest.TestCase):
    def test_aggregate_fit_can_exceed_one_simd(self):
        h=Hardware.load('superpod_compute/data/bw1100.json')
        m=Mapping(336,464,16,1,1,2,2,4096,1,2)
        original=GemmModel.capacity(m,h)
        self.assertTrue(original['valid'])
        split=dataclasses.replace(h,register_partitions_per_core=4,register_bytes_per_partition=192*1024)
        result=GemmModel.capacity(m,split)
        self.assertFalse(result['valid'])
        self.assertIn('register-partition',result['rejections'])
        self.assertEqual(max(result['register_bytes_per_partition']),203776)
    def test_balanced_resident_tile_fits(self):
        h=dataclasses.replace(Hardware.load('superpod_compute/data/bw1100.json'),register_partitions_per_core=4,register_bytes_per_partition=192*1024)
        result=GemmModel.capacity(Mapping(256,256,32,1,1,2,2),h)
        self.assertTrue(result['valid']);self.assertEqual(len(result['register_bytes_per_partition']),4)
    def test_unspecified_partition_preserves_serialized_hardware(self):
        h=Hardware.load('superpod_compute/data/bw1100.json')
        self.assertNotIn('register_partitions_per_core',h.to_dict())
        self.assertEqual(Hardware(**h.to_dict()),h)

if __name__=='__main__':unittest.main()
