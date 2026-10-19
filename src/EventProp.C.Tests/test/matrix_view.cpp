/**
 * @file matrix_view.cpp
 * @brief Tests for MatrixView / MutableMatrixView helpers (matrixview.inc).
 */

#include <gtest/gtest.h>
#include "matrixview.inc"
#include <vector>
#include <cstring>

// ──────────────────────────────────────────────────────────────
// mv() — immutable view construction
// ──────────────────────────────────────────────────────────────

TEST(MatrixView, ConstructorSetsFields) {
    double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    MatrixView v = mv(data, 2, 3);

    EXPECT_EQ(v.data, data);
    EXPECT_EQ(v.rows, 2);
    EXPECT_EQ(v.cols, 3);
}

// ──────────────────────────────────────────────────────────────
// mv_row() — row pointer arithmetic (immutable)
// ──────────────────────────────────────────────────────────────

TEST(MatrixViewRow, PointsToCorrectOffset) {
    double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    MatrixView v = mv(data, 2, 3);

    EXPECT_EQ(mv_row(v, 0), data + 0);
    EXPECT_EQ(mv_row(v, 1), data + 3);
}

TEST(MatrixViewRow, NonSquareMatrix) {
    double data[10] = {}; // 2x5
    MatrixView v = mv(data, 2, 5);

    EXPECT_EQ(mv_row(v, 0), data + 0);
    EXPECT_EQ(mv_row(v, 1), data + 5);
}

// ──────────────────────────────────────────────────────────────
// mv_at() — element access (immutable)
// ──────────────────────────────────────────────────────────────

TEST(MatrixViewAt, ReadsCorrectValues) {
    // Row-major: [a b c; d e f]
    double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    MatrixView v = mv(data, 2, 3);

    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 2), 3.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 1, 0), 4.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 1, 1), 5.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 1, 2), 6.0);
}

TEST(MatrixViewAt, SingleElement) {
    double data[1] = {42.0};
    MatrixView v = mv(data, 1, 1);

    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 0), 42.0);
}

TEST(MatrixViewAt, SingleRow) {
    double data[3] = {1.0, 2.0, 3.0};
    MatrixView v = mv(data, 1, 3);

    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 2), 3.0);
}

TEST(MatrixViewAt, SingleColumn) {
    double data[3] = {1.0, 2.0, 3.0};
    MatrixView v = mv(data, 3, 1);

    EXPECT_DOUBLE_EQ(*mv_at(v, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 1, 0), 2.0);
    EXPECT_DOUBLE_EQ(*mv_at(v, 2, 0), 3.0);
}

TEST(MatrixViewAt, NoAliasingInNonSquare) {
    // 3x4 matrix — ensure row-major stride doesn't alias elements
    double data[12] = {};
    for (int i = 0; i < 12; ++i) data[i] = static_cast<double>(i);

    MatrixView v = mv(data, 3, 4);

    // Each element should be at a unique address
    const double* p00 = mv_at(v, 0, 0);
    const double* p12 = mv_at(v, 1, 2);
    EXPECT_NE(p00, p12);
    EXPECT_DOUBLE_EQ(*p00, 0.0);
    EXPECT_DOUBLE_EQ(*p12, 6.0); // row 1 * 4 + col 2 = 6
}

// ──────────────────────────────────────────────────────────────
// mv_mut() — mutable view construction
// ──────────────────────────────────────────────────────────────

TEST(MutableMatrixView, ConstructorSetsFields) {
    double data[6] = {};
    MutableMatrixView v = mv_mut(data, 2, 3);

    EXPECT_EQ(v.data, data);
    EXPECT_EQ(v.rows, 2);
    EXPECT_EQ(v.cols, 3);
}

// ──────────────────────────────────────────────────────────────
// mv_row_mut() — row pointer arithmetic (mutable)
// ──────────────────────────────────────────────────────────────

TEST(MutableMatrixViewRow, PointsToCorrectOffset) {
    double data[6] = {}; // 2x3
    MutableMatrixView v = mv_mut(data, 2, 3);

    EXPECT_EQ(mv_row_mut(v, 0), data + 0);
    EXPECT_EQ(mv_row_mut(v, 1), data + 3);
}

// ──────────────────────────────────────────────────────────────
// mv_at_mut() — mutable element access
// ──────────────────────────────────────────────────────────────

TEST(MutableMatrixViewAt, WriteThroughModifiesUnderlyingArray) {
    double data[6] = {}; // 2x3
    MutableMatrixView v = mv_mut(data, 2, 3);

    *mv_at_mut(v, 0, 1) = 7.0;
    EXPECT_DOUBLE_EQ(data[1], 7.0);

    *mv_at_mut(v, 1, 2) = 9.0;
    EXPECT_DOUBLE_EQ(data[5], 9.0);
}

TEST(MutableMatrixViewAt, WritesDoNotAliasOtherElements) {
    double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    MutableMatrixView v = mv_mut(data, 2, 3);

    *mv_at_mut(v, 0, 0) = 100.0;

    EXPECT_DOUBLE_EQ(*mv_at_mut(v, 0, 0), 100.0); // changed
    EXPECT_DOUBLE_EQ(*mv_at_mut(v, 0, 1), 2.0);   // unchanged
    EXPECT_DOUBLE_EQ(*mv_at_mut(v, 1, 0), 4.0);   // unchanged
}

TEST(MutableMatrixViewAt, SingleElement) {
    double data[1] = {0.0};
    MutableMatrixView v = mv_mut(data, 1, 1);

    *mv_at_mut(v, 0, 0) = 42.0;
    EXPECT_DOUBLE_EQ(*mv_at_mut(v, 0, 0), 42.0);
}

// ──────────────────────────────────────────────────────────────
// Consistency: mutable and immutable views share the same layout
// ──────────────────────────────────────────────────────────────

TEST(MatrixViewConsistency, MutableAndImmutableMatch) {
    double data[12] = {}; // 3x4
    for (int i = 0; i < 12; ++i) data[i] = static_cast<double>(i);

    MatrixView imm = mv(data, 3, 4);
    MutableMatrixView mut = mv_mut(data, 3, 4);

    // All element pointers should match between views
    for (int r = 0; r < 3; ++r) {
        EXPECT_EQ(mv_row(imm, r), mv_row_mut(mut, r));
        for (int c = 0; c < 4; ++c) {
            EXPECT_EQ(mv_at(imm, r, c), mv_at_mut(mut, r, c));
        }
    }
}
