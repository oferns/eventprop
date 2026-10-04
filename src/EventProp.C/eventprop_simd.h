#ifndef EVENTPROP_SIMD_H
#define EVENTPROP_SIMD_H

#ifdef __cplusplus
extern "C" {
#endif

int simd_vec_width(void);
int simd_available(void);

void fma_row_simd(double* dest, const double* src, const double* weights, double scalar, int n);
void scale_inplace_simd(double* span, const double scalar, const int n);
void vadd_scaled_simd(double* dest, const double* src, const double scalar, const int n);
void vsub_simd(double* dest, const double* a, const double* b, int n);
double dot_product_simd(const double* a, const double* b, const int n);

#ifdef __cplusplus
}
#endif

#endif
