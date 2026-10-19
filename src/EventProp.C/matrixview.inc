/* ── MatrixView helpers (included, not compiled standalone) ─────── */
#ifndef MATRIXVIEW_INC
#define MATRIXVIEW_INC

typedef struct MatrixView {
    const double* data;
    int rows;
    int cols;   /* row stride (always == n_cols for dense matrices) */
} MatrixView;

static inline MatrixView mv(const double* data, int rows, int cols)
{
    MatrixView v;
    v.data = data; v.rows = rows; v.cols = cols;
    return v;
}

/** Pointer to row i (row-major: base + i * cols). */
static inline const double* mv_row(const MatrixView m, int i)
{ return m.data + i * m.cols; }

/** Element access: &m[row][col]. */
static inline const double* mv_at(const MatrixView m, int row, int col)
{ return &m.data[row * m.cols + col]; }

/** Mutable view (like Eigen::Ref<RowMatrixXd>). */
typedef struct MutableMatrixView {
    double* data;
    int rows;
    int cols;
} MutableMatrixView;

static inline MutableMatrixView mv_mut(double* data, int rows, int cols)
{
    MutableMatrixView v;
    v.data = data; v.rows = rows; v.cols = cols;
    return v;
}

static inline double* mv_row_mut(MutableMatrixView m, int i)
{ return m.data + i * m.cols; }

/** Mutable element access: &m[row][col]. */
static inline double* mv_at_mut(MutableMatrixView m, int row, int col)
{ return &m.data[row * m.cols + col]; }
#endif // MATRIXVIEW_INC
