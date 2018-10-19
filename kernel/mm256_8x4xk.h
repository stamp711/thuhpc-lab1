#pragma once
#include <immintrin.h>

#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

#define AVX2_LOAD_A(ymm, i, k) (ymm = _mm256_load_pd((A) + (i) + (k)*lda))
#define AVX2_LOAD_C(ymm, i, j) (ymm = _mm256_load_pd((C) + (i) + (j)*ldc))
#define AVX2_BROADCAST_B(ymm, k, j) (ymm = _mm256_broadcast_sd((B) + (k) + (j)*ldb))
#define AVX2_FMADD(c, a, b) (c = _mm256_fmadd_pd(a, b, c))
#define AVX2_STORE_C(ymm, i, j) (_mm256_store_pd((C) + (i) + (j)*ldc, ymm))

inline void kernel_mm256_8x4xk(int lda, int ldb, int ldc, int kc, const double *restrict A, const double *restrict B, double *restrict C)
{
  __assume_aligned(A, 32);

  // use 14 (of 16) ymm registers
  // C is 8 x 4
  register __m256d c_ymm_00, c_ymm_01, c_ymm_02, c_ymm_03;
  register __m256d c_ymm_40, c_ymm_41, c_ymm_42, c_ymm_43;
  // for each step in loop, A is 8 x 1
  register __m256d a_ymm_00, a_ymm_40;
  // for each step in loop, B is 1 x 4
  register __m256d b_val_00, b_val_01, b_val_02, b_val_03;

  // load C
  AVX2_LOAD_C(c_ymm_00, 0, 0);
  AVX2_LOAD_C(c_ymm_01, 0, 1);
  AVX2_LOAD_C(c_ymm_02, 0, 2);
  AVX2_LOAD_C(c_ymm_03, 0, 3);
  AVX2_LOAD_C(c_ymm_40, 4, 0);
  AVX2_LOAD_C(c_ymm_41, 4, 1);
  AVX2_LOAD_C(c_ymm_42, 4, 2);
  AVX2_LOAD_C(c_ymm_43, 4, 3);

  for (int k = 0; k < kc; k++)
  {
    // load col k from A
    AVX2_LOAD_A(a_ymm_00, 0, k);
    AVX2_LOAD_A(a_ymm_40, 4, k);

    // broadcast every value of row k from B
    AVX2_BROADCAST_B(b_val_00, k, 0);
    AVX2_BROADCAST_B(b_val_01, k, 1);
    AVX2_BROADCAST_B(b_val_02, k, 2);
    AVX2_BROADCAST_B(b_val_03, k, 3);

    // perform AVX2 FMA
    AVX2_FMADD(c_ymm_00, a_ymm_00, b_val_00);
    AVX2_FMADD(c_ymm_01, a_ymm_00, b_val_01);
    AVX2_FMADD(c_ymm_02, a_ymm_00, b_val_02);
    AVX2_FMADD(c_ymm_03, a_ymm_00, b_val_03);
    AVX2_FMADD(c_ymm_40, a_ymm_40, b_val_00);
    AVX2_FMADD(c_ymm_41, a_ymm_40, b_val_01);
    AVX2_FMADD(c_ymm_42, a_ymm_40, b_val_02);
    AVX2_FMADD(c_ymm_43, a_ymm_40, b_val_03);
  }
  // store results
  AVX2_STORE_C(c_ymm_00, 0, 0);
  AVX2_STORE_C(c_ymm_01, 0, 1);
  AVX2_STORE_C(c_ymm_02, 0, 2);
  AVX2_STORE_C(c_ymm_03, 0, 3);
  AVX2_STORE_C(c_ymm_40, 4, 0);
  AVX2_STORE_C(c_ymm_41, 4, 1);
  AVX2_STORE_C(c_ymm_42, 4, 2);
  AVX2_STORE_C(c_ymm_43, 4, 3);
}
