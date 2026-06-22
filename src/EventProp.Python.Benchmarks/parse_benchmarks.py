#!/usr/bin/env python3
"""Parse pytest-benchmark output into clean comparison tables."""

import subprocess, re, sys

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'test_benchmarks.py',
     '--benchmark-min-rounds=1', '--benchmark-max-time=0.15',
     '--benchmark-columns=min,max,mean,stddev', '--tb=no', '-q'],
    capture_output=True, text=True, timeout=300
)

# Parse benchmark table lines
data = {}
for line in result.stdout.split('\n'):
    if not line.startswith('test_'):
        continue
    # Extract name and mean value (5th numeric field after min,max,mean,stddev)
    parts = line.split()
    if len(parts) < 9:
        continue
    name = parts[0]
    # Mean is the 3rd number (index 4-5 in parts: "8.7625" "(1.0)")
    try:
        mean_val = float(re.sub(r'[,\s]', '', parts[4]))
        data[name] = mean_val
    except (ValueError, IndexError):
        pass

print(f'Parsed {len(data)} benchmarks')

# Define categories and extract comparisons
categories = [
    ('LIF Forward', 'lif_forward'),
    ('LI Forward', 'li_forward'),
    ('TTFS Forward', 'ttfs_forward'),
    ('VMax Forward', 'vmax_forward'),
    ('LIF Backward', 'lif_backward'),
    ('LI Backward', 'li_backward'),
    ('TTFS Backward', 'ttfs_backward'),
    ('VMax Backward', 'vmax_backward'),
]

sizes = ['Tiny', 'Small', 'Medium']
patterns_list = ['uniform', 'poisson', 'burst']

print()
print('=' * 95)
print('PYTHON BENCHMARK RESULTS - pybind11 vs Cython (mean us, lower is better)')
print('=' * 95)

for cat_name, cat_key in categories:
    print()
    print(f'--- {cat_name} ---')
    header = '{:<8} {:<10} {:>12} {:>12} {:>8}'.format('Size', 'Pattern', 'pybind11 us', 'Cython us', 'Ratio')
    print(header)
    print('-' * len(header))

    for size in sizes:
        for pat in patterns_list:
            py_key = f'test_{cat_key}_pybind11[{pat}-{size}]'
            cy_key = f'test_{cat_key}_cython[{pat}-{size}]'

            if py_key in data and cy_key in data:
                py_val = data[py_key]
                cy_val = data[cy_key]
                ratio = '{:.2f}x'.format(py_val/cy_val)
                winner = 'Cython!' if cy_val < py_val else 'pybind11'
                print('{:<8} {:<10} {:>12.1f} {:>12.1f} {:>7s}  ({})'.format(
                    size, pat, py_val, cy_val, ratio, winner))
            elif py_key in data:
                print('{:<8} {:<10} {:>12.1f}   SKIPPED'.format(size, pat, data[py_key]))

print()
print('=' * 95)
