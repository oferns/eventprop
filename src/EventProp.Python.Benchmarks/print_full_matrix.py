"""Full matrix comparison across all backends and sizes."""
import json, re, glob

# .NET sweep data
dotnet_sweep = {}
for f in glob.glob('../EventProp.Benchmarks/BenchmarkDotNet.Artifacts/results/*Sweep*-report-full-compressed.json'):
    data = json.load(open(f))
    for b in data.get('Benchmarks', []):
        display = b.get('DisplayInfo','')
        method = b.get('Method','')
        stats = b.get('Statistics')
        if not stats: continue
        mean_us = stats['Mean'] / 1e3
        m = re.match(r'Bench_TrainingLoopSweep_(\w+)_(\d+)', display)
        if m:
            size, spikes = m.groups()
            backend = 'Managed' if method == 'Managed' else 'Native'
            dotnet_sweep[(size, spikes, backend)] = mean_us

# Python sweep data
py_sweep = {}
for entry in json.load(open('results_sweep.json')).get('benchmarks', []):
    name = entry['name']
    stats = entry.get('stats')
    if not stats: continue
    mean_us = stats.get('mean', 0) * 1e6
    m = re.match(r'test_(pybind11|cython)_sweep\[([0-9]+)-(\w+)\]', name)
    if m:
        backend, spikes, size = m.groups()
        py_sweep[(size, spikes, backend)] = mean_us

# Native C/C++ data
def parse_native(fp):
    d = {}
    try:
        with open(fp) as f:
            for line in f:
                m2 = re.match(r'(\S+)\s+\S+ns\s+(\d+(?:\.\d+)?)\s+ns', line)
                if m2:
                    d[m2.group(1).replace('/real_time','')] = float(m2.group(2))/1000
    except FileNotFoundError: pass
    return d

c_data = parse_native('../EventProp.Native.Benchmarks/build/Release/bench_c_output.txt')
cpp_data = parse_native('../EventProp.Native.Benchmarks/build/Release/bench_cpp_output.txt')

print('=' * 85)
print('FULL MATRIX: All backends, all sizes (training step, uniform)')
print('=' * 85)

for size in ['Tiny', 'Small', 'Medium', 'Large']:
    print(f'\n--- {size} ---')
    
    # .NET sweep data
    for spikes_key in ['10','100','500','1000']:
        managed = dotnet_sweep.get((size, spikes_key, 'Managed'))
        native = dotnet_sweep.get((size, spikes_key, 'Native'))
        if managed and native:
            print(f'  Managed {spikes_key:>4s}spk: {managed:>9.1f}us | Native P/Invoke: {native:>9.1f}us ({native/managed:.2f}x)')
    
    # Python sweep data
    for spikes_key in ['10','100','500','1000']:
        pyb = py_sweep.get((size, spikes_key, 'pybind11'))
        cyt = py_sweep.get((size, spikes_key, 'cython'))
        if pyb and cyt:
            print(f'  pybind11 {spikes_key:>4s}spk: {pyb:>9.1f}us | Cython:       {cyt:>9.1f}us ({cyt/pyb:.2f}x)')
    
    # Native C/C++ (fixed sizes, uniform)
    c_ft = c_data.get(f'FullTrainingStep/{size}/uniform/default')
    cpp_ft = cpp_data.get(f'FullTrainingStep/{size}/uniform/default')
    if c_ft and cpp_ft:
        print(f'  Native C: {c_ft:>9.1f}us | Native C++: {cpp_ft:>9.1f}us ({cpp_ft/c_ft:.2f}x)')

print()
print('=' * 85)
print('P/Invoke overhead trend (Native / Managed ratio):')
for size in ['Tiny', 'Small', 'Medium', 'Large']:
    for spikes_key in ['10','100','500','1000']:
        managed = dotnet_sweep.get((size, spikes_key, 'Managed'))
        native = dotnet_sweep.get((size, spikes_key, 'Native'))
        if managed and native:
            print(f'  {size:>6s}/{spikes_key:>4s}spk: {native/managed:.2f}x ({native-managed:+.1f}us)')

print()
print('=' * 85)
print('Cython overhead trend (Cython / pybind11 ratio):')
for size in ['Tiny', 'Small', 'Medium', 'Large']:
    for spikes_key in ['10','100','500','1000']:
        pyb = py_sweep.get((size, spikes_key, 'pybind11'))
        cyt = py_sweep.get((size, spikes_key, 'cython'))
        if pyb and cyt:
            print(f'  {size:>6s}/{spikes_key:>4s}spk: {cyt/pyb:.2f}x ({cyt-pyb:+.1f}us)')
