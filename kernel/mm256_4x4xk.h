#pragma once
#include <immintrin.h>

#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

#define double4 __m256d

inline void kernel_mm256_4x4xk(int lda, int ldb, int ldc, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  register double4 c_ymm_00, c_ymm_01, c_ymm_02, c_ymm_03;

  c_ymm_00 = _mm256_setzero_pd();
  c_ymm_01 = _mm256_setzero_pd();
  c_ymm_02 = _mm256_setzero_pd();
  c_ymm_03 = _mm256_setzero_pd();

  for (int k = 0; k < K; k++)
  {
    double4 a_ymm_0k;
    double4 b_val_k0, b_val_k1, b_val_k2, b_val_k3;

    a_ymm_0k = _mm256_load_pd(&A(4 * 0, k));
    b_val_k0 = _mm256_broadcast_sd(&B(k, 0));
    c_ymm_00 = _mm256_fmadd_pd(a_ymm_0k, b_val_k0, c_ymm_00);

    b_val_k1 = _mm256_broadcast_sd(&B(k, 1));
    c_ymm_01 = _mm256_fmadd_pd(a_ymm_0k, b_val_k1, c_ymm_01);

    b_val_k2 = _mm256_broadcast_sd(&B(k, 2));
    c_ymm_02 = _mm256_fmadd_pd(a_ymm_0k, b_val_k2, c_ymm_02);

    b_val_k3 = _mm256_broadcast_sd(&B(k, 3));
    c_ymm_03 = _mm256_fmadd_pd(a_ymm_0k, b_val_k3, c_ymm_03);
  }

#define _mm256_load_add_store_pd_reg(ymm, addr, tmp) \
  double4 tmp;                                       \
  tmp = _mm256_load_pd((addr));                      \
  tmp = _mm256_add_pd((ymm), tmp);                   \
  _mm256_store_pd((addr), (tmp))

  _mm256_load_add_store_pd_reg(c_ymm_00, &C(0, 0), t_ymm_00);
  _mm256_load_add_store_pd_reg(c_ymm_01, &C(0, 1), t_ymm_01);
  _mm256_load_add_store_pd_reg(c_ymm_02, &C(0, 2), t_ymm_02);
  _mm256_load_add_store_pd_reg(c_ymm_03, &C(0, 3), t_ymm_03);
}
