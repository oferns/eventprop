/**
 * @file test_main.cpp
 * @brief Google Test entry point for EventProp C library.
 */

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
