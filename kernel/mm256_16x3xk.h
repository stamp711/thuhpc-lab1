#pragma once
#include <immintrin.h>

#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

#define regsA 4
#define regsB 3
#define double4 __m256d

inline __attribute__((always_inline))
void kernel_mm256_16x3xk(int lda, int ldb, int ldc, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  register double4 c_ymm_00, c_ymm_01, c_ymm_02;
  register double4 c_ymm_40, c_ymm_41, c_ymm_42;
  register double4 c_ymm_80, c_ymm_81, c_ymm_82;
  register double4 c_ymm_a0, c_ymm_a1, c_ymm_a2;

  c_ymm_00 = _mm256_setzero_pd();
  c_ymm_01 = _mm256_setzero_pd();
  c_ymm_02 = _mm256_setzero_pd();
  c_ymm_40 = _mm256_setzero_pd();
  c_ymm_41 = _mm256_setzero_pd();
  c_ymm_42 = _mm256_setzero_pd();
  c_ymm_80 = _mm256_setzero_pd();
  c_ymm_81 = _mm256_setzero_pd();
  c_ymm_82 = _mm256_setzero_pd();
  c_ymm_a0 = _mm256_setzero_pd();
  c_ymm_a1 = _mm256_setzero_pd();
  c_ymm_a2 = _mm256_setzero_pd();

  for (int k = 0; k < K; k++)
  {
    double4 a_ymm_0k, a_ymm_4k, a_ymm_8k, a_ymm_ak;
    double4 b_val_k0, b_val_k1, b_val_k2;

    a_ymm_0k = _mm256_load_pd(&A(4 * 0, k));
    b_val_k0 = _mm256_broadcast_sd(&B(k, 0));
    c_ymm_00 = _mm256_fmadd_pd(a_ymm_0k, b_val_k0, c_ymm_00);

    b_val_k1 = _mm256_broadcast_sd(&B(k, 1));
    c_ymm_01 = _mm256_fmadd_pd(a_ymm_0k, b_val_k1, c_ymm_01);

    b_val_k2 = _mm256_broadcast_sd(&B(k, 2));
    c_ymm_02 = _mm256_fmadd_pd(a_ymm_0k, b_val_k2, c_ymm_02);

    a_ymm_4k = _mm256_load_pd(&A(4 * 1, k));
    c_ymm_40 = _mm256_fmadd_pd(a_ymm_4k, b_val_k0, c_ymm_40);
    c_ymm_41 = _mm256_fmadd_pd(a_ymm_4k, b_val_k1, c_ymm_41);
    c_ymm_42 = _mm256_fmadd_pd(a_ymm_4k, b_val_k2, c_ymm_42);

    a_ymm_8k = _mm256_load_pd(&A(4 * 2, k));
    c_ymm_80 = _mm256_fmadd_pd(a_ymm_8k, b_val_k0, c_ymm_80);
    c_ymm_81 = _mm256_fmadd_pd(a_ymm_8k, b_val_k1, c_ymm_81);
    c_ymm_82 = _mm256_fmadd_pd(a_ymm_8k, b_val_k2, c_ymm_82);

    a_ymm_ak = _mm256_load_pd(&A(4 * 3, k));
    c_ymm_a0 = _mm256_fmadd_pd(a_ymm_ak, b_val_k0, c_ymm_a0);
    c_ymm_a1 = _mm256_fmadd_pd(a_ymm_ak, b_val_k1, c_ymm_a1);
    c_ymm_a2 = _mm256_fmadd_pd(a_ymm_ak, b_val_k2, c_ymm_a2);
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

  _mm256_load_add_store_pd_reg(c_ymm_01, &C(0, 1), t_ymm_01);
  _mm256_load_add_store_pd_reg(c_ymm_41, &C(4, 1), t_ymm_41);
  _mm256_load_add_store_pd_reg(c_ymm_81, &C(8, 1), t_ymm_81);
  _mm256_load_add_store_pd_reg(c_ymm_a1, &C(12, 1), t_ymm_a1);

  _mm256_load_add_store_pd_reg(c_ymm_02, &C(0, 2), t_ymm_02);
  _mm256_load_add_store_pd_reg(c_ymm_42, &C(4, 2), t_ymm_42);
  _mm256_load_add_store_pd_reg(c_ymm_82, &C(8, 2), t_ymm_82);
  _mm256_load_add_store_pd_reg(c_ymm_a2, &C(12, 2), t_ymm_a2);
}
