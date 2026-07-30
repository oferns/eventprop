"""Test backward pass through LIFLayer using native Cython bindings."""

import numpy as np
from eventprop_c.lif_layer import LIFLayer, LIFLayerParameters
from eventprop_c.spikes import SpikesVector, Spikes

np.random.seed(42)
n_in, n_out = 10, 5

params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=0.02, tau_syn=0.005, v_th=1.0)
layer = LIFLayer(params)

input_spikes = []
for _ in range(3):
    n_spikes = np.random.randint(5, 15)
    times = np.sort(np.random.rand(n_spikes) * 0.1)
    sources = np.random.randint(0, n_in, size=n_spikes).astype(np.int32)
    input_spikes.append(Spikes(times, sources))

input_batch = SpikesVector(input_spikes)
output = layer.forward(input_batch)
print(f'Output batch size: {len(output)}')

# Set errors on output spikes (required before backward pass)
for i in range(len(output)):
    for j in range(output[i].n_spikes):
        output[i].set_error(j, np.random.rand())
print('Errors set OK')

# Call backward directly through the native binding
from eventprop_c._native import backward_spikes_batch_accumulate
result = backward_spikes_batch_accumulate(layer.post_batch._batch_output)
print(f'Result grad shape: {result.grad.shape}')
print(f'Rows: {result.rows}, Cols: {result.cols}')

# Now test the full layer.backward() flow
layer2 = LIFLayer(params)
input_batch2 = SpikesVector(input_spikes)
output2 = layer2.forward(input_batch2)
for i in range(len(output2)):
    for j in range(output2[i].n_spikes):
        output2[i].set_error(j, np.random.rand())

print('Calling layer.backward()...')
layer2.backward()
print(f'Gradient shape: {layer2.gradient.shape}')
print('Done!')
