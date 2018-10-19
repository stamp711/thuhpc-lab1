#pragma once
#include <immintrin.h>

#define double4 __m256d

#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

#define _mm256_load_add_store_pd(ymm, addr, tmp) \
  {                                              \
    tmp = _mm256_load_pd((addr));                \
    tmp = _mm256_add_pd((ymm), tmp);             \
    _mm256_store_pd((addr), (tmp));              \
  }

#define generate_kernel(function, M, N)                                                                        \
  inline void function(int lda, int ldb, int ldc, int K, const double *A, const double *B, double *restrict C) \
  {                                                                                                            \
    double4 c_ymm[M][N];                                                                                       \
                                                                                                               \
    for (int i = 0; i < M; i++)                                                                                \
      for (int j = 0; j < N; j++)                                                                              \
        c_ymm[i][j] = _mm256_setzero_pd();                                                                     \
                                                                                                               \
    for (int k = 0; k < K; k++)                                                                                \
    {                                                                                                          \
      double4 a_ymm[M];                                                                                        \
      double4 b_val[N];                                                                                        \
                                                                                                               \
      for (int j = 0; j < N; j++)                                                                              \
      {                                                                                                        \
        b_val[j] = _mm256_broadcast_sd(&B(k, j));                                                              \
        for (int i = 0; i < M; i++)                                                                            \
        {                                                                                                      \
          a_ymm[i] = _mm256_load_pd(&A(4 * i, k));                                                             \
          c_ymm[i][j] = _mm256_fmadd_pd(a_ymm[i], b_val[j], c_ymm[i][j]);                                      \
        }                                                                                                      \
      }                                                                                                        \
    }                                                                                                          \
                                                                                                               \
    double4 tmp;                                                                                               \
    for (int j = 0; j < N; j++)                                                                                \
      for (int i = 0; i < M; i++)                                                                              \
        _mm256_load_add_store_pd(c_ymm[i][j], &C(4 * i, j), tmp);                                              \
  }
