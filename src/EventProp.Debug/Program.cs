using System;
using CW.EventProp;

int nPost = 3, nPre = 2;
double[] w = [20.0, 0.01, 0.01, 0.01, 0.01, 20.0];
double[] times = [0.01, 0.02];
int[] sources = [0, 1];

// C# implementation
var buffers = new ScratchBuffers();
buffers.InitPostSpikeBuffers(nPost, 8);
var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

Console.WriteLine($"C# spikes: {csResult.NSpikes}");
for (int i = 0; i < csResult.NSpikes; i++)
    Console.WriteLine($"  [{i}] t={csResult.Times[i]:F8}, src={csResult.Sources[i]}, cur={csResult.Currents[i]:F4}");

// C++ via P/Invoke
var cppTimes = new double[10];
var cppSources = new int[10];
var cppCurrents = new double[10];
int nCppSpikes = CxxNative.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, cppTimes, cppSources, cppCurrents);

Console.WriteLine($"\nC++ spikes: {nCppSpikes}");
for (int i = 0; i < nCppSpikes; i++)
    Console.WriteLine($"  [{i}] t={cppTimes[i]:F8}, src={cppSources[i]}, cur={cppCurrents[i]:F4}");
