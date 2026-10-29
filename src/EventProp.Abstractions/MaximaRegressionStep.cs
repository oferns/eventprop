namespace CW.EventProp;

/// <summary>
/// Batch outputs of <c>LILayer.Regression</c> — the continuous readout
/// <c>y_hat = bias + Σ a[n] · maxima[n]</c> over the maxima from the last <c>Forward()</c>,
/// with the loss written into each sample's maxima error buffers.
/// </summary>
/// <remarks>
/// <para>
/// The carrier lives here rather than beside either per-sample result because both backends
/// return it: the native primitive is <c>Native.Loss.RegressionResult</c> and the managed one is
/// <c>RegressionLoss.RegressionResult</c>, and those two are the pair held to each other by the
/// parity tests. Naming the batch step the same thing would make every parity assertion read
/// ambiguously.
/// </para>
/// <para>
/// <see cref="DecoderGrad"/> and <see cref="BiasGrad"/> follow the library contract: they are
/// <b>sums</b> over the batch and they <b>accumulate</b> into the buffers the caller passed (when
/// omitted, fresh ones are allocated). Nothing divides by a batch size and nothing zeroes —
/// clearing is the caller's job, exactly as <see cref="Layer.ZeroGrad"/> is for the weights.
/// </para>
/// </remarks>
/// <param name="Predictions">y_hat per sample.</param>
/// <param name="Losses">Loss per sample.</param>
/// <param name="Skipped">Neurons per sample whose maxima were non-finite: no peak, so no readout
/// signal. A valid datum rather than an error — a whole sample can be skipped.</param>
/// <param name="DecoderGrad">dL/da, summed over the batch, length ≥ the layer's neuron count.</param>
/// <param name="BiasGrad">dL/db, summed over the batch, as a length-1 array.</param>
public readonly record struct MaximaRegressionStep(
    double[] Predictions,
    double[] Losses,
    int[] Skipped,
    double[] DecoderGrad,
    double[] BiasGrad);
