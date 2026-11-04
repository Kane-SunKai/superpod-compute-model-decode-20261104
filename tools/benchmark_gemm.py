"""Single-GPU GEMM measurements and independent model comparison.

measure needs a working CUDA/HIP PyTorch installation. It can run standalone.
predict/compare need this repository but do not import PyTorch or need a GPU.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = {
    'dtype': 'fp16',
    'cases': [
        {'name': 'regular', 'M': 3072, 'N': 2304, 'K': 1024},
        {'name': 'non_multiple', 'M': 130, 'N': 97, 'K': 270},
        {'name': 'single_row', 'M': 1, 'N': 8192, 'K': 8192},
    ],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def checked_cases(config, dtype_override=None):
    dtype = dtype_override or config.get('dtype', 'fp16')
    if dtype not in ('fp16', 'bf16'):
        raise ValueError('dtype must be fp16 or bf16')
    cases = config.get('cases', [])
    if not cases:
        raise ValueError('At least one case is required')
    output, names = [], set()
    for case in cases:
        name = case.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Each case needs a unique nonempty name')
        names.add(name)
        for dim in ('M', 'N', 'K'):
            value = case.get(dim)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name}: {dim} must be a positive integer')
        output.append({key: case[key] for key in ('name', 'M', 'N', 'K')})
    return dtype, output


def summarize(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Timing samples must be finite and positive')
    ordered = sorted(values)
    # Nearest-rank P90. Raw samples are retained for alternative statistics.
    return {'min': ordered[0], 'p50': statistics.median(ordered),
            'p90': ordered[math.ceil(0.9 * len(ordered)) - 1],
            'mean': statistics.mean(ordered),
            'stdev': statistics.pstdev(ordered), 'count': len(ordered)}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def environment_base():
    return {'utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def configure_precision(torch):
    """Record what this installed PyTorch can actually request, not assumed ISA."""
    flags, warnings = {}, []
    settings = ('allow_fp16_reduced_precision_reduction',
                'allow_bf16_reduced_precision_reduction', 'allow_fp16_accumulation')
    for name in settings:
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, False)
            flags[name] = getattr(torch.backends.cuda.matmul, name)
        else:
            warnings.append(f'{name} not exposed by this PyTorch build')
    warnings.append('Library chooses its own kernel/tiling; these switches do not '
                    'enforce the model mapping or universally disable split-K.')
    return flags, warnings


def correctness_sample(torch, a, b, c, dtype):
    # Sample up to 8 evenly spaced rows/columns, and sum over ALL K on CPU FP64.
    rows = sorted({i * (a.shape[0] - 1) // 7 for i in range(8)})
    cols = sorted({i * (b.shape[1] - 1) // 7 for i in range(8)})
    aa = a[rows, :].cpu().double()
    bb = b[:, cols].cpu().double()
    actual = c[rows, :][:, cols].cpu().double()
    expected = aa @ bb
    atol, rtol = (0.002, 0.02) if dtype == 'fp16' else (0.02, 0.05)
    return {'passed': bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
            'method': 'sample rows/columns, full K, CPU FP64 reference',
            'sample_rows': rows, 'sample_columns': cols, 'atol': atol, 'rtol': rtol,
            'max_absolute_error': float((actual - expected).abs().max()),
            'relative_l2_error': float((actual - expected).norm() /
                                       expected.norm().clamp_min(1e-30))}


def measure(args):
    dtype, cases = checked_cases(read_json(args.cases) if args.cases else DEFAULT_CASES,
                                 args.dtype)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Use the real machine Python environment with CUDA/HIP '
                           'PyTorch installed; this script will not install it.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access a CUDA/HIP GPU; no CPU fallback is used.')
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError('Device index is outside the visible device list')
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')
    torch.manual_seed(args.seed)
    flags, warnings = configure_precision(torch)
    prop = torch.cuda.get_device_properties(device)
    properties = {'name': prop.name, 'total_memory_bytes': prop.total_memory,
                  'multiprocessor_count': getattr(prop, 'multi_processor_count', None),
                  'properties_repr': str(prop)}
    l2_size = getattr(prop, 'L2_cache_size', None)
    if l2_size is None:
        l2_size = getattr(prop, 'l2_cache_size', None)
    properties['reported_l2_bytes'] = l2_size
    env = {**environment_base(), 'torch_version': torch.__version__,
           'cuda_build': torch.version.cuda, 'hip_build': torch.version.hip,
           'visible_device_index': args.device, 'device': properties,
           'precision_flags': flags,
           'environment_variables': {name: os.environ.get(name) for name in
               ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG')}}
    modes = ['warm', 'evict'] if args.cache_mode == 'both' else [args.cache_mode]
    eviction_bytes = args.flush_mib * 1024**2 if 'evict' in modes else 0
    if l2_size and eviction_bytes and eviction_bytes < 2 * l2_size:
        warnings.append('Eviction buffer is smaller than 2x the reported L2; increase --flush-mib.')
    result = {'format': 'superpod-gemm-measurement-v1', 'environment': env,
              'dtype': dtype, 'operation': 'C=AB; contiguous row-major; no bias; no autograd',
              'settings': {'warmup': args.warmup, 'iterations': args.iterations,
                           'seed': args.seed, 'cache_modes': modes,
                           'eviction_bytes': eviction_bytes},
              'warnings': warnings, 'cases': [],
              'timing_scope': {
                  'gpu_event_us': 'Event interval around ONE torch.mm(out=C); includes device '
                                  'work and possible dispatch gaps; excludes allocation, '
                                  'random initialization, correctness check and eviction.',
                  'host_observed_us': 'Host wall time for torch.mm + end-event record/wait; '
                                      'includes Python/runtime/synchronization/instrumentation.',
                  'evict': 'Heuristic cache thrashing before each call, not guaranteed cache '
                           'invalidation. Does not reproduce cold-each-wave inside the model.'}}
    torch_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize lazily-created event objects before collecting samples.
    start.record(); end.record(); end.synchronize()
    with torch.inference_mode():
        evict = torch.empty(eviction_bytes, dtype=torch.uint8, device=device) if eviction_bytes else None
        for case in cases:
            print(f"Measuring {case['name']}: {case['M']}x{case['N']}x{case['K']} {dtype}", flush=True)
            a = torch.randn((case['M'], case['K']), dtype=torch_dtype, device=device)
            b = torch.randn((case['K'], case['N']), dtype=torch_dtype, device=device)
            a.mul_(0.1); b.mul_(0.1)
            c = torch.empty((case['M'], case['N']), dtype=torch_dtype, device=device)
            torch.mm(a, b, out=c)
            torch.cuda.synchronize(device)
            check = correctness_sample(torch, a, b, c, dtype)
            record = {**case, 'correctness': check, 'modes': {}}
            result['cases'].append(record)
            if not check['passed']:
                write_json(args.output, result)
                raise RuntimeError(f"{case['name']}: sampled correctness failed; saved diagnostic JSON")
            for mode in modes:
                for _ in range(args.warmup):
                    torch.mm(a, b, out=c)
                torch.cuda.synchronize(device)
                gpu, host = [], []
                for iteration in range(args.iterations):
                    if mode == 'evict':
                        evict.fill_(iteration % 251)
                        torch.cuda.synchronize(device)
                    start.record()
                    t0 = time.perf_counter_ns()
                    torch.mm(a, b, out=c)
                    end.record()
                    end.synchronize()
                    host.append((time.perf_counter_ns() - t0) / 1000)
                    gpu.append(start.elapsed_time(end) * 1000)  # API gives milliseconds
                record['modes'][mode] = {
                    'gpu_event_us': summarize(gpu), 'host_observed_us': summarize(host),
                    'gpu_event_samples_us': gpu, 'host_observed_samples_us': host,
                    'effective_tflops': 2 * case['M'] * case['N'] * case['K'] /
                                        (statistics.median(gpu) * 1e6)}
                print(f"  {mode}: GPU event p50={statistics.median(gpu):.3f} us; "
                      f"host p50={statistics.median(host):.3f} us", flush=True)
                write_json(args.output, result)
            del a, b, c
    result['completed'] = True
    write_json(args.output, result)
    print(f'Saved {args.output}')


def predict_cases(dtype, cases, args):
    sys.path.insert(0, str(ROOT))
    from superpod_compute import GemmRequest, Hardware, Policy, predict
    hardware = Hardware.load(args.hardware)
    policy = Policy(**read_json(args.policy)) if args.policy else Policy()
    predictions = []
    for case in cases:
        request = GemmRequest(case['M'], case['N'], case['K'], dtype=dtype)
        result = predict(request, hardware, policy=policy, backend=args.backend,
                         cache_path=args.cache)
        best = result['best']
        predictions.append({**case, 'model_total_us': best['duration_us'],
                            'configured_launch_us': hardware.kernel_launch_s * 1e6,
                            'model_without_configured_launch_us':
                                (best['duration_s'] - hardware.kernel_launch_s) * 1e6,
                            'prediction': result})
    return {'format': 'superpod-gemm-prediction-v1', 'environment': environment_base(),
            'dtype': dtype, 'hardware': hardware.to_dict(), 'policy': policy.to_dict(),
            'cases': predictions}


def predict_command(args):
    dtype, cases = checked_cases(read_json(args.cases) if args.cases else DEFAULT_CASES,
                                 args.dtype)
    report = predict_cases(dtype, cases, args)
    write_json(args.output, report)
    for case in report['cases']:
        print(f"{case['name']}: model total={case['model_total_us']:.3f} us; "
              f"without configured launch={case['model_without_configured_launch_us']:.3f} us")


def comparison_rows(measured, prediction):
    if measured['dtype'] != prediction['dtype']:
        raise ValueError('Measured and predicted dtypes do not match')
    lookup = {case['name']: case for case in prediction['cases']}
    rows = []
    for real in measured['cases']:
        sim = lookup[real['name']]
        if any(real[dim] != sim[dim] for dim in ('M', 'N', 'K')):
            raise ValueError('Measured and predicted shapes do not match')
        if not real['correctness']['passed']:
            raise ValueError('Cannot compare failed correctness results')
        for mode, timing in real['modes'].items():
            event = timing['gpu_event_us']['p50']
            if not math.isfinite(event) or event <= 0:
                raise ValueError('Invalid measured event time')
            rows.append({**{k: real[k] for k in ('name', 'M', 'N', 'K')},
                         'dtype': measured['dtype'], 'cache_mode': mode,
                         'gpu_event_p50_us': event,
                         'host_observed_p50_us': timing['host_observed_us']['p50'],
                         'model_total_us': sim['model_total_us'],
                         'configured_launch_us': sim['configured_launch_us'],
                         'model_without_configured_launch_us': sim['model_without_configured_launch_us'],
                         'raw_total_vs_event_difference_pct': 100 * (sim['model_total_us'] / event - 1),
                         'raw_no_launch_vs_event_difference_pct':
                             100 * (sim['model_without_configured_launch_us'] / event - 1)})
    return rows


def compare_command(args):
    measured = read_json(args.measured)
    if measured.get('format') != 'superpod-gemm-measurement-v1' or not measured.get('completed'):
        raise ValueError('Expected a completed measurement report from this script')
    dtype, cases = checked_cases(measured)
    prediction = predict_cases(dtype, cases, args)
    rows = comparison_rows(measured, prediction)
    report = {'format': 'superpod-gemm-comparison-v1', 'measurement': measured,
              'model': prediction, 'rows': rows,
              'interpretation': [
                  'Differences are descriptive, not a matched-kernel accuracy certification.',
                  'The real library chooses its own mapping, instruction path and cache reuse.',
                  'Event timing is not equivalent to model total including configured launch; '
                  'the no-launch column is a diagnostic view, not an automatic correction.',
                  'Eviction is only before the whole real GEMM; model assumes cold at each wave.',
                  'Verify the model hardware profile matches the measured device; neither '
                  'device name nor detected properties auto-calibrate the profile.']}
    write_json(args.output, report)
    csv_path = Path(args.output).with_suffix('.csv')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('case / cache | GPU event p50 us | model total us | model without configured launch us')
    for row in rows:
        print(f"{row['name']} / {row['cache_mode']} | {row['gpu_event_p50_us']:.3f} | "
              f"{row['model_total_us']:.3f} | {row['model_without_configured_launch_us']:.3f}")
    print(f'Saved {args.output} and {csv_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    measure_parser = commands.add_parser('measure', help='Measure on one real GPU with PyTorch')
    predict_parser = commands.add_parser('predict', help='Run the model without a GPU')
    compare_parser = commands.add_parser('compare', help='Predict the measured cases and write JSON/CSV')
    for p in (measure_parser, predict_parser):
        p.add_argument('--cases', type=Path, help='JSON; defaults to the three built-in cases')
        p.add_argument('--dtype', choices=('fp16', 'bf16'), help='Override JSON dtype')
    for p in (predict_parser, compare_parser):
        p.add_argument('--hardware', type=Path, help='Model hardware JSON; default bundled A100')
        p.add_argument('--policy', type=Path)
        p.add_argument('--backend', choices=('auto', 'array', 'peak', 'mmac', 'mac-throughput'), default='auto')
        p.add_argument('--cache', type=Path, help='Optional model search cache SQLite file')
    measure_parser.add_argument('--device', type=int, default=0)
    measure_parser.add_argument('--warmup', type=positive_int, default=30)
    measure_parser.add_argument('--iterations', type=positive_int, default=100)
    measure_parser.add_argument('--seed', type=int, default=2026)
    measure_parser.add_argument('--cache-mode', choices=('warm', 'evict', 'both'), default='both')
    measure_parser.add_argument('--flush-mib', type=positive_int, default=256)
    measure_parser.add_argument('--output', type=Path, required=True)
    predict_parser.add_argument('--output', type=Path, required=True)
    compare_parser.add_argument('--measured', type=Path, required=True)
    compare_parser.add_argument('--output', type=Path, required=True)
    measure_parser.set_defaults(func=measure)
    predict_parser.set_defaults(func=predict_command)
    compare_parser.set_defaults(func=compare_command)
    args = parser.parse_args(argv)
    if args.command == 'compare' and (args.output.suffix.lower() != '.json' or
                                    args.output.resolve() == args.measured.resolve()):
        parser.error('Comparison --output must be a new .json path, different from --measured')
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
