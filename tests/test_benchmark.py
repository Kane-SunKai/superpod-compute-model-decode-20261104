"""Benchmark configuration, timing units, and report matching (no fake GPU results)."""
import copy
import unittest

from tools.benchmark_gemm import (
    DEFAULT_CASES, checked_cases, comparison_rows, summarize,
)


class BenchmarkContractTests(unittest.TestCase):
    def test_default_cases_match_model_validation(self):
        dtype, cases = checked_cases(DEFAULT_CASES)
        self.assertEqual(dtype, 'fp16')
        self.assertEqual([(c['M'], c['N'], c['K']) for c in cases],
                         [(3072, 2304, 1024), (130, 97, 270), (1, 8192, 8192)])

    def test_invalid_dimensions_and_duplicate_names_rejected(self):
        for invalid in (0, -1, True, 1.5, '3'):
            config = copy.deepcopy(DEFAULT_CASES)
            config['cases'][0]['M'] = invalid
            with self.assertRaises(ValueError):
                checked_cases(config)
        config = copy.deepcopy(DEFAULT_CASES)
        config['cases'][1]['name'] = config['cases'][0]['name']
        with self.assertRaises(ValueError):
            checked_cases(config)

    def test_dtype_override_does_not_mutate_config(self):
        self.assertEqual(checked_cases(DEFAULT_CASES, 'bf16')[0], 'bf16')
        self.assertEqual(DEFAULT_CASES['dtype'], 'fp16')
        with self.assertRaises(ValueError):
            checked_cases(DEFAULT_CASES, 'fp8')

    def test_statistics_retain_microsecond_units(self):
        result = summarize([1, 2, 3, 4, 5])
        self.assertEqual(result['p50'], 3)
        self.assertEqual(result['p90'], 5)
        for samples in ([], [0], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                summarize(samples)

    def fixture(self):
        # Artificial numbers only test arithmetic; never emitted as a measurement.
        shape = {'name': 'fixture', 'M': 16, 'N': 32, 'K': 64}
        measured = {'dtype': 'fp16', 'cases': [{**shape,
            'correctness': {'passed': True}, 'modes': {'evict': {
                'gpu_event_us': {'p50': 10}, 'host_observed_us': {'p50': 15}}}}]}
        predicted = {'dtype': 'fp16', 'cases': [{**shape, 'model_total_us': 30,
            'configured_launch_us': 21, 'model_without_configured_launch_us': 9}]}
        return measured, predicted

    def test_comparison_preserves_both_scopes(self):
        measured, predicted = self.fixture()
        row = comparison_rows(measured, predicted)[0]
        self.assertEqual(row['raw_total_vs_event_difference_pct'], 200)
        self.assertAlmostEqual(row['raw_no_launch_vs_event_difference_pct'], -10)
        self.assertEqual(row['configured_launch_us'], 21)

    def test_comparison_rejects_mismatched_shape_dtype_and_failed_check(self):
        for change in ('shape', 'dtype', 'check'):
            measured, predicted = self.fixture()
            if change == 'shape':
                predicted['cases'][0]['K'] = 32
            elif change == 'dtype':
                predicted['dtype'] = 'bf16'
            else:
                measured['cases'][0]['correctness']['passed'] = False
            with self.assertRaises(ValueError):
                comparison_rows(measured, predicted)


if __name__ == '__main__':
    unittest.main()
