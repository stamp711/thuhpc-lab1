#pragma once
#include <immintrin.h>

#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

#define double4 __m256d

inline void kernel_mm256_16x1xk(int lda, int ldb, int ldc, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  register double4 c_ymm_00;
  register double4 c_ymm_40;
  register double4 c_ymm_80;
  register double4 c_ymm_a0;

  c_ymm_00 = _mm256_setzero_pd();
  c_ymm_40 = _mm256_setzero_pd();
  c_ymm_80 = _mm256_setzero_pd();
  c_ymm_a0 = _mm256_setzero_pd();

  for (int k = 0; k < K; k++)
  {
    double4 a_ymm_0k, a_ymm_4k, a_ymm_8k, a_ymm_ak;
    double4 b_val_k0;

    a_ymm_0k = _mm256_load_pd(&A(4 * 0, k));
    b_val_k0 = _mm256_broadcast_sd(&B(k, 0));
    c_ymm_00 = _mm256_fmadd_pd(a_ymm_0k, b_val_k0, c_ymm_00);

    a_ymm_4k = _mm256_load_pd(&A(4 * 1, k));
    c_ymm_40 = _mm256_fmadd_pd(a_ymm_4k, b_val_k0, c_ymm_40);

    a_ymm_8k = _mm256_load_pd(&A(4 * 2, k));
    c_ymm_80 = _mm256_fmadd_pd(a_ymm_8k, b_val_k0, c_ymm_80);

    a_ymm_ak = _mm256_load_pd(&A(4 * 3, k));
    c_ymm_a0 = _mm256_fmadd_pd(a_ymm_ak, b_val_k0, c_ymm_a0);
  }

#define _mm256_load_add_store_pd_reg(ymm, addr, tmp) \
  double4 tmp;                                       \
  tmp = _mm256_load_pd((addr));                      \
  tmp = _mm256_add_pd((ymm), tmp);                   \
  _mm256_store_pd((addr), (tmp))

  _mm256_load_add_store_pd_reg(c_ymm_00, &C(0, 0), t_ymm_00);
  _mm256_load_add_store_pd_reg(c_ymm_40, &C(4, 0), t_ymm_40);
  _mm256_load_add_store_pd_reg(c_ymm_80, &C(8, 0), t_ymm_80);
  _mm256_load_add_store_pd_reg(c_ymm_a0, &C(12, 0), t_ymm_a0);
}