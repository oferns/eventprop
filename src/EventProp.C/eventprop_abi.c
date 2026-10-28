/**
 * @file eventprop_abi.c
 * @brief ABI self-check: the version and struct sizes this build was compiled with.
 *
 * Every consumer of this library COMPILES the public struct layouts into its own
 * binary: the Cython extension (`_native.pyx` cdef extern blocks), the .NET
 * P/Invoke layer (`EventProp.Native`, [StructLayout] structs) and the C++
 * consumers. A header change that moves a field offset is therefore invisible at
 * link time — the consumer keeps reading the offsets it compiled, against a library
 * whose fields moved. It does not crash on load; it reads plausible-looking garbage
 * from the wrong offsets.
 *
 * Measured, not theoretical: a `_native.cp311-win_amd64.pyd` built against July
 * headers, loaded against a library rebuilt from October headers, produced an access
 * violation inside `spikes.py __getitem__` via `loss_layer._find_first_spikes` —
 * 40 minutes into a benchmark run, in a frame with no hint that the cause was a
 * header/binary offset mismatch. Nothing in the build checked that the extension was
 * newer than the headers it was compiled from.
 *
 * So each consumer checks the match at load, and a mismatch is a hard error naming
 * both values:
 *
 *   - `eventprop_abi_version()` must equal `EVENTPROP_ABI_VERSION` as compiled from
 *     the consumer's own copy of eventprop.h.
 *   - `eventprop_abi_struct_size(name)` must equal the consumer's own `sizeof` for
 *     every struct it embeds by value.
 *
 * The size check is the mechanical half: it needs no bump discipline, and it fails
 * on exactly the thing that broke (offsets), which is why both are exported.
 */

#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"

#include <string.h>

int eventprop_abi_version(void) {
    return EVENTPROP_ABI_VERSION;
}

/* name/size pairs for every public struct a consumer embeds by value: the five
 * output structs in eventprop.h, the input struct, the batch types in
 * eventprop_batch.h and the diagnostic types in eventprop_diag.h. Stringified with
 * #name so the table cannot drift from the types it measures — renaming a typedef
 * renames its key. */
#define EVENTPROP_SIZE_ENTRY(name) { #name, sizeof(name) }

size_t eventprop_abi_struct_size(const char* struct_name) {
    static const struct {
        const char* name;
        size_t size;
    } sizes[] = {
        EVENTPROP_SIZE_ENTRY(EventPropInput),
        EVENTPROP_SIZE_ENTRY(SpikeOutput),
        EVENTPROP_SIZE_ENTRY(SumsOutput),
        EVENTPROP_SIZE_ENTRY(MaximaOutput),
        EVENTPROP_SIZE_ENTRY(BackwardOutput),
        EVENTPROP_SIZE_ENTRY(SpikesBatch),
        EVENTPROP_SIZE_ENTRY(MaximaBatch),
        EVENTPROP_SIZE_ENTRY(BackwardsBatch),
        EVENTPROP_SIZE_ENTRY(GradientOutput),
        EVENTPROP_SIZE_ENTRY(DropoutOutput),
        EVENTPROP_SIZE_ENTRY(DropoutBatch),
        EVENTPROP_SIZE_ENTRY(AccumulatedBackwardResult),
        EVENTPROP_SIZE_ENTRY(VoltageTraceInput),
        EVENTPROP_SIZE_ENTRY(VoltageTraceOutput),
        EVENTPROP_SIZE_ENTRY(LambdaITraceInput),
        EVENTPROP_SIZE_ENTRY(LambdaITraceOutput),
    };
    const size_t count = sizeof(sizes) / sizeof(sizes[0]);

    if (!struct_name) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(sizes[i].name, struct_name) == 0) {
            return sizes[i].size;
        }
    }
    /* A consumer asking about a struct this library does not know is itself a
     * mismatch, so 0 is the answer that makes the caller's check fail loudly. */
    return 0;
}
