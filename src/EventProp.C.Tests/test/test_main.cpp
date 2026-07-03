/**
 * @file test_main.cpp
 * @brief Google Test scaffold for EventProp C library.
 */

#include <gtest/gtest.h>

TEST(ScaffoldTest, GTestIsWorking) {
    EXPECT_TRUE(true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
