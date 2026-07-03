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

# Set errors
for i in range(len(output)):
    for j in range(output[i].n_spikes):
        output[i].set_error(j, np.random.rand())
print('Errors set OK')

# Get flat buffers from input
times, sources, n_per_sample = SpikesVector._extract_flat_buffers(layer.input_batch)
print(f'Input flat: times={len(times)}, sources={len(sources)}')

# Get errors from output
errors = layer.post_batch._batch_output.get_flat_errors()
print(f'Errors shape: {errors.shape}')

# Call backward directly with debug
from eventprop_c._native import backward_spikes_batch_accumulate
result = backward_spikes_batch_accumulate(
    times, sources, n_per_sample,
    layer.post_batch._batch_output, errors,
)
print(f'Result grad shape: {result.grad.shape}')

# Now test the full layer.backward()
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
