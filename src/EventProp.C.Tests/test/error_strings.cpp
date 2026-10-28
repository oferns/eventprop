/**
 * @file error_strings.cpp
 * @brief Smoke tests for eventprop_error_string — every defined code must map
 *        to a non-NULL, unique message; out-of-range codes map to "Unknown error".
 */

#include <gtest/gtest.h>
#include "eventprop.h"

#include <set>
#include <string>

TEST(ErrorStrings, EveryDefinedCodeHasUniqueNonNullMessage) {
    // Upper bound = highest defined enumerator (EVENTPROP_NONFINITE_PREDICTION).
    // A code past the end must fall through to "Unknown error", so the loop
    // also proves the table is dense (no accidental gaps).
    int last_known = 0;
    std::set<std::string> seen;

    for (int code = 0; code <= 256; ++code) {
        const char* msg = eventprop_error_string(code);
        EXPECT_NE(msg, nullptr);

        const std::string s(msg);
        if (s == "Unknown error") {
            // First gap ends the table; everything after must also be unknown.
            last_known = code - 1;
            break;
        }
        EXPECT_EQ(seen.insert(s).second, true)
            << "Duplicate message for code " << code << ": " << s;
    }

    // The table must be dense: no code inside the range may be unknown.
    EXPECT_GE(last_known, EVENTPROP_NONFINITE_PREDICTION)
        << "A code below EVENTPROP_NONFINITE_PREDICTION returned \"Unknown error\"";
}

TEST(ErrorStrings, SpecificMessagesAreCorrect) {
    // Guard against the historical typo and misleading wording.
    EXPECT_STREQ(eventprop_error_string(EVENTPROP_OK), "OK");
    EXPECT_STREQ(eventprop_error_string(EVENTPROP_NEGATIVE_SPIKES), "Negative spike count");
    EXPECT_STREQ(eventprop_error_string(EVENTPROP_UNSORTED_SPIKE_TIMES),
                 "Spike times are not sorted in ascending order");
    EXPECT_STREQ(eventprop_error_string(EVENTPROP_ALLOC_BWD_CURRENTS),
                 "Allocation failed: backward-pass post-spike currents");
    EXPECT_STREQ(eventprop_error_string(EVENTPROP_NONFINITE_SPIKE_TIME),
                 "Spike time is not finite (NaN)");
}

TEST(ErrorStrings, OutOfRangeCodesReturnUnknown) {
    EXPECT_STREQ(eventprop_error_string(-1), "Unknown error");
    EXPECT_STREQ(eventprop_error_string(100000), "Unknown error");
}
