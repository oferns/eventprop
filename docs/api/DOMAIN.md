# EventProp cross-implementation API — the domain

**Status: stage 1 of 3.** This file owns the *terms* and the *actions*. Stage 2 derives
`docs/api/openapi.json` from it; stage 3 implements that document twice — once over the Python
bindings (C library, Eigen reference) and once over the .NET ones (managed `CW.EventProp`, P/Invoke
over `eventprop_c_api`). Nothing else in the repository defines this vocabulary.

One owner per fact, deliberately:

| Question | Owner |
|---|---|
| What a word means, what an action is, what is out of scope | this file |
| What a request/response looks like on the wire | `docs/api/openapi.json` (stage 2) |
| Struct layouts, error codes and their numeric values | `src/EventProp.C/eventprop.h` |
| What a specific binary can actually do | that implementation's `describeImplementation` response |

## 1. What this is for

EventProp computes **exact gradients through spike times** for spiking neural networks, and this
repository ships *several independent implementations of that computation*. The interesting claim is
not "there is a C library" — it is that a managed C# implementation, a P/Invoke binding, a Cython
binding and an Eigen reference all agree on the same inputs, to a stated tolerance, including on
where they refuse.

So the contract exists to make that claim **checkable from outside the test runners**. If it cannot
be expressed here, it does not belong on the wire.

**Not in this contract:** figure and plot shapes, Streamlit state, the Python pickle checkpoint
format (a bridge concern), Python-side training artefacts, and anything a single stack can only do
because of its own convenience layer.

## 2. Ubiquitous language

These words come from the code that already exists (`eventprop.h`, `CW.EventProp`,
`EventProp.Abstractions`, the Cython layer). A schema named something else for the same thing is a
second vocabulary, and second vocabularies are how two implementations stop talking about the same
object.

| Term | Wire name | Meaning |
|---|---|---|
| **Network** | `network` | The layer stack and its neuron parameters (`tauMem`, `tauSyn`, `vTh`) — the *shape*, without weights. |
| **Parameters** | `parameters` | One trained weight/bias set for a network. Value, not identity; identity belongs to a ParameterSet. |
| **ParameterSet** | `parameterSetId` | A named, server-held Parameters with provenance: source checkpoint, run seed, snapshot epoch, ABI of the file's producer. |
| **Stimulus** | `stimulus` | Pre-encoding numbers: a point in the plane, a digit's pixels. Never a spike train. |
| **Encoding** | `encoding` | A named, versioned rule `Stimulus → SpikeTrain` (e.g. `ttfs-linear-v1`). The version is load-bearing, see §6.2. |
| **SpikeTrain** | `spikeTrain` | `times` + `sources`, ascending, non-negative. The one data type the whole domain turns into other things. |
| **SpikeTrainBatch** | `spikeTrainBatch` | A list of SpikeTrains. Trains are carried explicitly — see §6.3. |
| **Maxima** | `maxima` | Per-neuron voltage maxima of a forward pass: a *continuous* quantity carried by the same spikes. |
| **Read-out** | `readout` | How a response becomes an answer: `first-spike` order, or a linear read-out over Maxima. |
| **ErrorSignal** | `errorSignal` | Per-output-spike error supplied by the caller. Owned by the caller, never allocated by the library. |
| **Loss** | `loss` | Which error function: `spike` (drives the adjoint from an ErrorSignal), `leastSquares`, `huber` with a δ. |
| **Target** | `target` | The continuous value a Loss is scored against. |
| **Adjoint** | `adjoint` | λ(t), the signal that carries error backwards through time to each threshold crossing. |
| **Gradient** | `gradient` | dLoss/dParameters — the product of the backward pass, not the process that produced it. |
| **Trace** | `trace` | An observation of one forward pass: membrane voltage, the adjoint, or a pointwise λ query at a time *t*. A Trace is an observation, never a step of a computation. |
| **Implementation** | `implementation` | One way of computing all of the above, identified by language, binding, ABI version and source revision. |
| **Capability** | `capabilities` | What that Implementation can do and — asymmetrically — whether it checks what it is given. |
| **Evaluation** | `evaluation` | Metrics for one Implementation over a PointSet, aggregated server-side. |
| **Agreement** | `agreement` | A bounded claim between two Implementations on the same inputs: tolerance, per-case expectations, verdict. |
| **Measurement** | `measurement` | A cost, carrying the same provenance as everything else, because a number without provenance is an anecdote. |

Two words are **banned** from the contract:

- **"backend"** — it already means "one of the two native bindings" (`eventweb/backends.py`) and
  "the server" in the same breath. Say **Implementation**.
- **"batch" as a type name** — the two existing bindings disagree about what a batch's length means
  (the Cython layer requires exact array lengths, the .NET side accepts "at least" spans and ignores
  trailing entries). The wire settles that argument instead of inheriting it, so the type is a
  `SpikeTrainBatch` of explicit trains, and no length convention is implied.

## 3. The compute surface is stateless

`f(Network, Parameters, Stimulus | SpikeTrain) → Response`. No action depends on a previous one,
there is no session to get into the wrong state, and nothing can drift because of call order. That
is a property of the algorithm, not a design preference, and the contract keeps it: every action is
a complete request.

Exactly two things are named on the server, and they are the two things that can silently differ
between stacks:

- **ParameterSet** — the only payload too large to repeat per request (a 784×350 float32 layer is
  ~1 MB), and the only thing whose *identity* a claim of agreement depends on. Naming it turns
  "which weights produced this?" from a footnote into a lookup. It is also where the Python→.NET
  weight bridge lands in the domain: the bridge materialises a ParameterSet, and the fact that it
  came from a Python pickle becomes part of its provenance rather than an accident of one stack.
- **PointSet** — the inputs. If each stack reads its own copy of a dataset, a divergence may be a
  data bug rather than a maths bug. Serving the same bytes to both Implementations is what makes an
  Agreement mean what it says.

## 4. The actions

Ten verbs. Everything else is a detail of one of these.

| # | Action | What it does | Returns |
|---|---|---|---|
| 1 | `describeImplementation` | Who am I: language, binding, ABI version, source revision, dirty flag — and what I can do. | Implementation, Capabilities |
| 2 | `listParameterSets` | Which trained networks this server can run, with provenance. | ParameterSet descriptors |
| 3 | `getPoints` | A slice of a named split. | PointSet |
| 4 | `encode` | Stimulus → SpikeTrain under a named Encoding. | SpikeTrainBatch, the scheme it used |
| 5 | `forward` | Run the network. Optionally with the Read-out applied. | SpikeTrainBatch, optional Maxima, optional read-out |
| 6 | `observe` | One Trace: voltage, adjoint, or pointwise λ at a time *t*. | Trace |
| 7 | `backward` | Loss + forward state → Gradient. Two input shapes because the two loss families genuinely differ: `spike` needs an ErrorSignal, `leastSquares`/`huber` need Targets plus a Read-out. | Gradient, read-out gradients where the loss has them |
| 8 | `evaluate` | Forward and score over a PointSet, aggregated **server-side**. | Evaluation |
| 9 | `agree` | Same request bytes to two Implementations, bounded comparison. | Agreement |
| 10 | `measure` | Cost of one of the above, same envelope. | Measurement |

Three consequences worth stating out loud:

- **`evaluate` aggregates rather than transports.** 10 000 points × 200 neurons is 2 million floats;
  as JSON that is tens of megabytes to move a computation that costs microseconds per sample. The
  contract offers tasks the size of the question, not primitives the size of the kernel — which is
  the same reason the Visualizer's pages ask for accuracy rather than for spike tensors.
- **`agree` is not implemented by either server.** A server that grades its own agreement with
  itself is not evidence. It lives in the harness or the BFF, as a *client of both*.
- **`observe` is separate from `forward` on purpose.** A Trace is why the app exists, and it is also
  the least portable surface: it is the one action where an Implementation may legitimately answer
  `unsupported` (see §5). Folding it into `forward` would force every implementation to produce
  diagnostics or lie about them.

## 5. Outcomes are not HTTP statuses

EventProp's distinguishing behaviour is that it **refuses**: unsorted spike times, an out-of-range
source, a non-finite target, a Huber δ that is neither positive-finite nor infinite. The numeric
codes and their meanings belong to `eventprop.h`; the contract carries their *names*.

Every action therefore returns one of three outcomes, and the schema must make all three
first-class:

```
computed   → data + envelope
refused    → { codeName }            e.g. UNSORTED_SPIKE_TIMES, INVALID_SOURCE,
                                     INVALID_HUBER_DELTA, NONFINITE_PREDICTION
unsupported→ { capability }          this Implementation cannot do this at all
```

And `agree` must be **expectation-aware**, because the implementations are not equally strict. An
unsorted spike list is rejected by the C library with `EVENTPROP_UNSORTED_SPIKE_TIMES` (60), while
the Eigen reference has the same requirement and does **not** check it — it returns plausible
garbage. A harness that demands "both refuse" reports a false divergence on its first run; a harness
that compares only numbers forgives a missing diagnostic, which is the defect the strictness exists
to catch. So each comparison case carries `expect: computed | refused(codeName)`, and a Capability
entry (`validatesInput: true | false`) states which side of that asymmetry an Implementation is on.

The general rule: **a divergence that is documented as a capability is a fact; the same divergence
unannounced is a bug.** The contract's job is to make the two distinguishable.

## 6. Recorded decisions

Each was a real fork. Reversing one is cheap only while there is one implementation of the contract.

1. **`train` is out of v1.** Single-step `backward` exists on every stack; a server-driven training
   loop does not exist at all on the .NET side (there is no trainer and no encoder there today), and
   putting training in the contract means .NET implements an optimiser *before* anyone has proved the
   forward and backward passes agree on real data. v1 proves the maths on real points; v2 can prove
   the learning.
2. **`encode` is inside the boundary**, named and versioned. Otherwise real-data parity belongs to
   whichever stack happened to encode the input, and a divergence between two encoder conventions
   (a `t_max`, a mirrored pair, a tie rule) would be reported as a maths disagreement. The version
   suffix is the whole mechanism: `ttfs-linear-v1` can never silently mean something else, which is
   the failure this repository has already paid for three times in its benchmark generators, where
   three copies of one generator must be kept in step by hand.
3. **A SpikeTrainBatch carries its trains explicitly** (a list of `times`/`sources` pairs) rather
   than flat arrays plus offsets. Self-describing wins over compact here: the existing bindings
   disagree about length semantics, offsets are where off-by-one bugs live, and this is a contract
   meant to expose disagreement rather than manufacture it. Payload size is controlled by §4's
   aggregation, not by cramming.
4. **`agree` belongs to a client of both stacks**, never to either server.
5. **Terms live here, wire shapes live in `openapi.json`.** Schemas name these terms; they do not
   redefine them. If a schema needs a new concept, this file gains a row first.
6. **`describeImplementation` is authoritative about capability; this file is a claim.** The
   capability expectations below were read off the code at stage 1 and are written as *expected*:

   | Capability | C library | Eigen reference | managed .NET | P/Invoke .NET |
   |---|---|---|---|---|
   | forward, backward | yes | yes | yes | yes |
   | Maxima read-out | yes | yes | yes | yes |
   | regression loss + read-out gradients | yes | **no** (no regression loss exists) | yes (independent implementation) | yes |
   | voltage Trace | yes | yes | yes (`GetVoltageTrace`) | yes |
   | adjoint Trace (λ) | yes | yes | **split**: `LIFLayer` implements it, `LILayer` inherits a base that throws `InvalidOperationException` | yes |
   | validates its input | yes | **no** | probe | yes |
   | encoder | no | in the reference *Python* layer, not the core | no | no |
   | trainer | no | in the reference Python layer | no | no |

   The remaining `probe` cells are deliberate: the contract exists partly to answer questions the
   test suite has not had to answer across stacks before. A server that reports a capability it does
   not have is a contract violation caught by stage 3's conformance test, not a documentation error.

   The split managed row is not a gap to apologise for — it is the clearest argument in this section.
   One stack, one layer type, one genuine `unsupported`, already expressed in that stack's own idiom.

**Every stack already refuses, in a different idiom.** A C error code; a managed
`InvalidOperationException`; a Cython `ValueError`; a pybind11 `RuntimeError`; and, in the Eigen
reference's case, silence. The three outcomes in §5 are the translation layer between those idioms,
so stage 2 must state the mapping — which native failure becomes `refused`, which becomes
`unsupported`, and what happens when a stack's failure arrives *late* (an `InvalidOperationException`
raised mid-request is a 500 unless the server recognizes it as a capability refusal). A client that
cannot tell "cannot do this" from "tried and failed" will eventually retry the unretryable.

## 7. The envelope

Every response carries it, because a claim of agreement between two binaries that cannot say which
binaries they were is unfalsifiable — the same reason every benchmark record in
`BENCHMARK_SCHEMA.md` carries host, revision and dirty flags.

```
meta: {
  implementation   : "c-cython" | "eigen-pybind11" | "managed-dotnet" | "pinvoke-dotnet" | …
  abiVersion       : integer         # the C library's ABI, bumped on any struct layout change
  sourceRevision   : string          # vcs_rev of the tree the binary came from
  sourceDirty      : boolean
  host             : string          # e.g. darwin-arm64 — numbers are not portable between them
  toolVersion      : string
  durationMs       : number
  encoding         : scheme name     # on every action that encoded anything
}
```

`abiVersion` is not decoration: consumers compile struct layouts into their own binaries, so a
mismatched library returns plausible garbage rather than failing. A server whose binding was built
against a different ABI must refuse, and must say so in `describeImplementation`.

## 8. Scope

**v1:** actions 1–8 and 10, one ParameterSet family (Yin-Yang), one PointSet, `agree` as a harness,
three outcomes, the envelope.

**v2, deliberately deferred, each with a reason:** `train` (§6.1); the MNIST ParameterSet (bigger
bridge, same loader); comparison of more than two Implementations at once (start with a pairwise
verdict — a three-way disagreement needs a different report than a two-way one); streaming/chunked
responses (only once a real payload forces it, not in anticipation).

## 9. Open questions for stage 2

- Do `forward` and `observe` share a request body, or does `observe` re-run the forward pass? Re-running
  is simpler and stateless; sharing is what the Visualizer actually wants. Decide with the first two
  operations written down.
- Numeric tolerance: per-action, or one band with exceptions? The regression loss is held to 1e-10
  today between managed and native; root-found spike times are not that tight. The spec needs a
  stated band per action, not a global "close enough".
- Where `pointwise λ at t` sits relative to `observe`: it is a third thing (a Trace *query* rather
  than a Trace *series*) and the first candidate for a second action.
- Units on the wire: seconds everywhere today in the C API, milliseconds in several plots. Seconds on
  the wire, named in the schema, and no conversion inside an Implementation.
