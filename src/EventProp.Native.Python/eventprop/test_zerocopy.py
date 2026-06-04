#!/usr/bin/env python3
"""Test zero-copy access with ctypes bindings (no numpy required)."""

import sys
import os

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eventprop import NativeApi as EventPropCtypes


def test_compute_spikes():
    """Test compute_spikes with zero-copy access."""
    
    # Create test data (small network)
    n_pre, n_post = 10, 5
    n_spikes = 20
    
    # High weights to trigger spikes
    w = [1.0] * (n_pre * n_post)
    
    # Uniform spike times
    times = [i / n_spikes for i in range(n_spikes)]
    
    # Sources cycle through pre neurons
    sources = [i % n_pre for i in range(n_spikes)]
    
    # Load library (provide path to your built library)
    lib_path = "/Users/ollie/repos/eventprop.net/src/EventProp.C/build/libeventprop_c_api.dylib"
    lib = EventPropCtypes(lib_path)
    
    # Compute spikes with zero-copy access
    output = lib.compute_spikes(w, times, sources)
    
    print(f"Output spikes: {output.n_spikes}")
    
    if output.n_spikes > 0:
        # Zero-copy access to spike times (no allocation!)
        print(f"First spike time: {output.times[0]:.6f}")
        print(f"Last spike time: {output.times[-1]:.6f}")
        
        # Access sources (no copy!)
        print(f"First source: {output.sources[0]}")
        
        # Iterate over spikes (zero-copy)
        print("First 5 spikes:")
        for i in range(min(5, output.n_spikes)):
            print(f"  Spike {i}: time={output.times[i]:.6f}, source={output.sources[i]}")
    
    # Cleanup is automatic on garbage collection
    
    return output.n_spikes > 0


def test_compute_sums():
    """Test compute_sums with zero-copy access."""
    
    n_pre, n_post = 10, 5
    n_spikes = 20
    
    w = [1.0] * (n_pre * n_post)
    times = [i / n_spikes for i in range(n_spikes)]
    sources = [i % n_pre for i in range(n_spikes)]
    
    lib_path = "/Users/ollie/repos/eventprop.net/src/EventProp.C/build/libeventprop_c_api.dylib"
    lib = EventPropCtypes(lib_path)
    
    # Compute sums with zero-copy access
    output = lib.compute_sums(w, times, sources)
    
    print(f"\nSums output: n_rows={output.n_rows}, n_post={output.n_post}")
    print(f"Sum0 length: {len(output.sum0)}")
    
    if len(output.sum0) > 0:
        print(f"First sum0 value: {output.sum0[0]:.6f}")
        print(f"First sum1 value: {output.sum1[0]:.6f}")
    
    return output.n_rows > 0


def test_compute_maxima():
    """Test compute_maxima with zero-copy access."""
    
    n_pre, n_post = 10, 5
    n_spikes = 20
    
    w = [1.0] * (n_pre * n_post)
    times = [i / n_spikes for i in range(n_spikes)]
    sources = [i % n_pre for i in range(n_spikes)]
    
    lib_path = "/Users/ollie/repos/eventprop.net/src/EventProp.C/build/libeventprop_c_api.dylib"
    lib = EventPropCtypes(lib_path)
    
    # Compute maxima with zero-copy access
    output = lib.compute_maxima(w, times, sources)
    
    print(f"\nMaxima output: n_post={output.n_post}")
    print(f"Times length: {len(output.times)}")
    
    if len(output.times) > 0:
        print(f"First maxima time: {output.times[0]:.6f}")
        print(f"First maxima value: {output.values[0]:.6f}")
    
    return output.n_post > 0


if __name__ == "__main__":
    print("Testing EventProp ctypes zero-copy bindings")
    print("=" * 50)
    
    try:
        result1 = test_compute_spikes()
        print(f"✓ compute_spikes test passed: {result1}")
    except Exception as e:
        print(f"✗ compute_spikes test failed: {e}")
    
    try:
        result2 = test_compute_sums()
        print(f"✓ compute_sums test passed: {result2}")
    except Exception as e:
        print(f"✗ compute_sums test failed: {e}")
    
    try:
        result3 = test_compute_maxima()
        print(f"✓ compute_maxima test passed: {result3}")
    except Exception as e:
        print(f"✗ compute_maxima test failed: {e}")
    
    print("\nAll tests completed!")
