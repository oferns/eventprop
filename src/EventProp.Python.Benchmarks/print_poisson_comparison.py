"""Compare uniform vs poisson patterns across all backends."""
import json, re, glob

def parse_native(file_path):
    data = {}
    try:
        with open(file_path) as f:
            for line in f:
                m = re.match(r'(\S+)\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+ns\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+ns', line)
                if m:
                    name = m.group(1).replace('/real_time', '')
                    cpu_ns = float(m.group(3))
                    data[name] = cpu_ns / 1000.0
    except FileNotFoundError:
        pass
    return data

c_small = parse_native('../EventProp.Native.Benchmarks/build/Release/bench_c_small.txt')
cpp_small = parse_native('../EventProp.Native.Benchmarks/build/Release/bench_cpp_small.txt')

# Python layer benchmarks (poisson, Small) from results_py.json
py_layer = {}
for entry in json.load(open('results_py.json')).get('benchmarks', []):
    name = entry['name']
    stats = entry.get('stats')
    if not stats: continue
    mean_us = stats.get('mean', 0) * 1e6
    m = re.match(r'test_(lif|li)_forward_(pybind11|cython)\[poisson-Small\]', name)
    if m:
        layer, backend = m.groups()
        py_layer[(f'{layer.upper()} Forward', backend, 'poisson')] = mean_us
    m = re.match(r'test_(lif|li)_backward_(pybind11|cython)\[poisson-Small\]', name)
    if m:
        layer, backend = m.groups()
        py_layer[(f'{layer.upper()} Backward', backend, 'poisson')] = mean_us

# Python sweep benchmarks (uniform, Small) from results_sweep.json
py_sweep = {}
for entry in json.load(open('results_sweep.json')).get('benchmarks', []):
    name = entry['name']
    stats = entry.get('stats')
    if not stats: continue
    mean_us = stats.get('mean', 0) * 1e6
    m = re.match(r'test_(pybind11|cython)_sweep\[([0-9]+)-Small\]', name)
    if m:
        backend, spikes = m.groups()
        py_sweep[(spikes, backend)] = mean_us

# .NET sweep benchmarks (uniform only)
dotnet_sweep = {}
for f in glob.glob('../EventProp.Benchmarks/BenchmarkDotNet.Artifacts/results/*Sweep*-report-full-compressed.json'):
    data = json.load(open(f))
    for b in data.get('Benchmarks', []):
        display = b.get('DisplayInfo','')
        method = b.get('Method','')
        stats = b.get('Statistics')
        if not stats: continue
        mean_us = stats['Mean'] / 1e3
        m = re.match(r'Bench_TrainingLoopSweep_Small_(\d+)', display)
        if m:
            spikes = m.group(1)
            backend = 'Managed' if method == 'Managed' else 'Native'
            dotnet_sweep[(spikes, backend)] = mean_us

print("=" * 90)
print("UNIFORM vs POISSON PATTERN COMPARISON - Small (50->20 neurons)")
print("=" * 90)
print()
print("Uniform: deterministic weights > threshold => zero output spikes")
print("Poisson: random timing => actual output spikes (realistic training)")
print()

# Native C/C++ comparison for poisson at Small size
print("[1] NATIVE C vs C++ (Small): Individual operations, uniform vs poisson")
print("-" * 70)
print(f"{'Operation':<24s} {'Pattern':>8s} | {'Native C':>10s} {'Native C++':>10s} {'C faster':>8s}")
print('-' * 70)
for op in ['ComputeSums', 'ComputeSpikes', 'BackwardSpikes']:
    for pat in ['uniform', 'poisson']:
        key = f'{op}/Small/{pat}/default'
        c_val = c_small.get(key, None)
        cpp_val = cpp_small.get(key, None)
        if c_val and cpp_val:
            print(f"{op:<24s} {pat:>8s} | {c_val:>10.1f}us {cpp_val:>10.1f}us {cpp_val/c_val:>7.1f}x")

print()
print("[2] PYTHON BACKENDS (Small, poisson): LIF layer benchmarks")
print("-" * 70)
for op in ['LIF Forward', 'LIF Backward']:
    print(f"\n  {op}:")
    for backend in ['pybind11', 'cython']:
        key = (op, backend, 'poisson')
        val = py_layer.get(key)
        if val:
            print(f"    {backend:<10s}: {val:>8.1f}us")

print()
print("[3] ALL BACKENDS - Full training step comparison")
print("-" * 70)
print("\n  (a) Uniform pattern, Small/100 spikes:")
native_c = c_small.get('FullTrainingStep/Small/uniform/default', None)
backends_uniform = [
    ('Managed .NET', dotnet_sweep.get(('100', 'Managed'))),
    ('Native P/Invoke', dotnet_sweep.get(('100', 'Native'))),
    ('Native C (raw)', native_c),
    ('pybind11', py_sweep.get(('100', 'pybind11'))),
    ('Cython', py_sweep.get(('100', 'cython'))),
]
min_val = min(v for _, v in backends_uniform if v is not None)
for name, val in backends_uniform:
    if val:
        print(f"    {name:<24s}: {val:>8.1f}us ({val/min_val:.1f}x)")

print("\n  (b) Poisson pattern, Small/100 spikes:")
native_c_fwd = c_small.get('ComputeSpikes/Small/poisson/default', None)
native_c_bwd = c_small.get('BackwardSpikes/Small/poisson/default', None)
if native_c_fwd and native_c_bwd:
    print(f"    Native C (raw, fwd+bwd): {native_c_fwd + native_c_bwd:>8.1f}us")

for backend in ['pybind11', 'cython']:
    fwd = py_layer.get((f'LIF Forward', backend, 'poisson'))
    bwd = py_layer.get((f'LIF Backward', backend, 'poisson'))
    if fwd and bwd:
        print(f"    {backend:<24s} (LIF fwd+bwd): {fwd + bwd:>8.1f}us")

print()
print("=" * 90)
print("KEY INSIGHT:")
print("-" * 70)
print("Poisson pattern produces actual output spikes, making BackwardSpikes")
print("significantly more expensive (25us C vs 395us C++ at Small).")
print("The C library's advantage grows from 3.6x (ComputeSpikes uniform)")
print("to 6.6x (poisson) and 15.6x (BackwardSpikes poisson).")
print("=" * 90)
