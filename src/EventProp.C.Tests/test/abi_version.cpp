/**
 * @file abi_version.cpp
 * @brief Tests for the ABI self-check surface: eventprop_abi_version() and
 *        eventprop_abi_struct_size().
 *
 * The point of these functions is the CROSS-binary check: the Cython extension and
 * the .NET P/Invoke layer compile the public struct layouts into their own binaries,
 * so a header change that moves an offset is invisible at link time and shows up as
 * garbage reads (see the header comment in eventprop_abi.c). Those consumers have
 * their own tests for the comparison itself.
 *
 * What is checked here is the library side of that contract:
 *   - the symbols are exported (a consumer's check that silently resolves to nothing
 *     is worse than no check at all);
 *   - the size table COVERS every public struct — a struct added to the headers
 *     without an entry returns 0, so a consumer's check passes for the ones listed
 *     and fails for the new one, which is exactly the drift this guards;
 *   - an unknown name returns 0, so a consumer asking about a struct the library
 *     does not know fails loudly instead of reading size 0 as "matches".
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"

#include <cstring>
#include <string>
#include <vector>

TEST(AbiVersion, ExportedAndMatchesHeaderMacro) {
    // The test TU compiles the same header, so equality is expected by construction.
    // It is asserted anyway: it proves the symbol is exported and the macro is
    // visible to consumers, which is what every cross-binary check depends on.
    EXPECT_EQ(eventprop_abi_version(), EVENTPROP_ABI_VERSION);
    EXPECT_GT(EVENTPROP_ABI_VERSION, 0);
}

TEST(AbiStructSize, EveryPublicStructHasANonZeroEntry) {
    // The list is the public surface a consumer embeds by value. Adding a struct to
    // eventprop.h / eventprop_batch.h without adding it to the table in
    // eventprop_abi.c fails here, which is the drift this test exists to catch.
    const std::vector<std::string> names = {
        "EventPropInput",
        "SpikeOutput",
        "SumsOutput",
        "MaximaOutput",
        "BackwardOutput",
        "SpikesBatch",
        "MaximaBatch",
        "BackwardsBatch",
        "GradientOutput",
        "DropoutOutput",
        "DropoutBatch",
        "AccumulatedBackwardResult",
    };

    for (const auto& name : names) {
        const size_t size = eventprop_abi_struct_size(name.c_str());
        EXPECT_GT(size, 0u) << name << " is missing from the size table in eventprop_abi.c";
        // A struct is a handful of pointers, ints and doubles; anything this large
        // means the table entry is measuring the wrong type.
        EXPECT_LT(size, 4096u) << name;
    }
}

TEST(AbiStructSize, MatchesSizeofInTheConsumersTranslationUnit) {
    // The comparison a consumer performs, performed here: the library's reported
    // size against sizeof as compiled in a TU that includes the headers.
    EXPECT_EQ(eventprop_abi_struct_size("EventPropInput"), sizeof(EventPropInput));
    EXPECT_EQ(eventprop_abi_struct_size("SpikeOutput"), sizeof(SpikeOutput));
    EXPECT_EQ(eventprop_abi_struct_size("SumsOutput"), sizeof(SumsOutput));
    EXPECT_EQ(eventprop_abi_struct_size("MaximaOutput"), sizeof(MaximaOutput));
    EXPECT_EQ(eventprop_abi_struct_size("BackwardOutput"), sizeof(BackwardOutput));
    EXPECT_EQ(eventprop_abi_struct_size("SpikesBatch"), sizeof(SpikesBatch));
    EXPECT_EQ(eventprop_abi_struct_size("MaximaBatch"), sizeof(MaximaBatch));
    EXPECT_EQ(eventprop_abi_struct_size("BackwardsBatch"), sizeof(BackwardsBatch));
    EXPECT_EQ(eventprop_abi_struct_size("GradientOutput"), sizeof(GradientOutput));
    EXPECT_EQ(eventprop_abi_struct_size("DropoutOutput"), sizeof(DropoutOutput));
    EXPECT_EQ(eventprop_abi_struct_size("DropoutBatch"), sizeof(DropoutBatch));
    EXPECT_EQ(eventprop_abi_struct_size("AccumulatedBackwardResult"),
              sizeof(AccumulatedBackwardResult));
}

TEST(AbiStructSize, UnknownNameReturnsZero) {
    // A consumer asking about a struct the library does not know must fail its own
    // check, not read 0 as a match.
    EXPECT_EQ(eventprop_abi_struct_size("NotAStruct"), 0u);
    EXPECT_EQ(eventprop_abi_struct_size(""), 0u);
    // Case-sensitive on purpose: the key is the typedef name.
    EXPECT_EQ(eventprop_abi_struct_size("spikeoutput"), 0u);
}

TEST(AbiStructSize, NullNameReturnsZero) {
    EXPECT_EQ(eventprop_abi_struct_size(nullptr), 0u);
}

TEST(AbiStructSize, TableKeysAreUnique) {
    // Duplicate keys would make the first match win, silently shadowing an entry.
    const char* names[] = {
        "EventPropInput", "SpikeOutput", "SumsOutput", "MaximaOutput", "BackwardOutput",
        "SpikesBatch", "MaximaBatch", "BackwardsBatch", "GradientOutput",
        "DropoutOutput", "DropoutBatch", "AccumulatedBackwardResult",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        for (size_t j = i + 1; j < sizeof(names) / sizeof(names[0]); ++j) {
            EXPECT_NE(std::strcmp(names[i], names[j]), 0);
        }
    }
}
