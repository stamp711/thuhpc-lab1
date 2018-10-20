#include "helper.h"
#include "kernel/naive.h"
#include "kernel/mm256_16x3xk.h"
#include "kernel/mm256_16x2xk.h"
#include "kernel/mm256_16x1xk.h"
#include "kernel/mm256_12x4xk.h"
#include "kernel/mm256_12x3xk.h"
#include "kernel/mm256_12x2xk.h"
#include "kernel/mm256_12x1xk.h"
#include "kernel/mm256_8x5xk.h"
#include "kernel/mm256_8x4xk.h"
#include "kernel/mm256_8x3xk.h"
#include "kernel/mm256_8x2xk.h"
#include "kernel/mm256_8x1xk.h"

#define DRIVER_USE_12
#define DRIVER_USE_8

const char *dgemm_desc = "Apricity's optimized dgemm.";

// Level 1 blocking
#define M1 512
#define N1 768
#define K1 512

// Level 2 blocking
#define M2 64
#define N2 768
#define K2 256

// Register blocking, i-j, K unchanged
#define Mc 16
#define Nc 3
#define Kc K
double Ai[Mc * K2] __attribute((aligned(32)));
#define Ai(k) (Ai[(k)*Mc])
inline void kernel_driver(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  const double *restrict block_A;
  const double *restrict block_B;
  double *restrict block_C;

  // Calculate block remainder for N
  int n = N % Nc;

  // Main blocking for M (16 x N)
  int i;
  for (i = 0; i <= M - Mc; i += Mc)
  {
    pack(Ai, Mc, &A(i, 0), lda, Mc, K);
    // Main blocking for N (16 x 3)
    int j;
    for (j = 0; j <= N - Nc; j += Nc)
    {
      block_B = &B(0, j);
      block_C = &C(i, j);
      kernel_mm256_16x3xk(Mc, ldb, ldc, K, Ai, block_B, block_C);
    } // End main blocking for N (16 x 3)
    // Deal with remainder 16 x n, n < 3
    if (n != 0)
    {
      block_B = &B(0, j);
      block_C = &C(i, j);
      if (n == 2)
        kernel_mm256_16x2xk(Mc, ldb, ldc, K, Ai, block_B, block_C);
      else
        kernel_mm256_16x1xk(Mc, ldb, ldc, K, Ai, block_B, block_C);
    } // End dealing with remadinder 16 x n
  }   // End main blocking for M

  // Calculate block remainder for M
  int m = M % Mc;

#ifdef DRIVER_USE_12
  // Deal with remainder m x N, m < 16
  if (m >= 12)
  {
    pack(Ai, 12, &A(i, 0), lda, 12, K);
    int j;
    for (j = 0; j <= N - 4; j += 4)
    {
      block_B = &B(0, j);
      block_C = &C(i, j);
      kernel_mm256_12x4xk(12, ldb, ldc, K, Ai, block_B, block_C);
    }
    n = N - j;
    if (n != 0)
    {
      block_B = &B(0, j);
      block_C = &C(i, j);
      if (n == 3)
        kernel_mm256_12x3xk(12, ldb, ldc, K, Ai, block_B, block_C);
      else if (n == 2)
        kernel_mm256_12x2xk(12, ldb, ldc, K, Ai, block_B, block_C);
      else
        kernel_mm256_12x1xk(12, ldb, ldc, K, Ai, block_B, block_C);
    }
    m -= 12;
    i += 12;
  }
#endif

#ifdef DRIVER_USE_8
  if (m >= 8)
  {
    pack(Ai, 8, &A(i, 0), lda, 8, K);
    int j;
    for (j = 0; j <= N - 5; j += 5)
    {
      block_B = &B(0, j);
      block_C = &C(i, j);
      kernel_mm256_8x5xk(8, ldb, ldc, K, Ai, block_B, block_C);
    }
    n = N - j;
    if (n != 0)
    {
      block_B = &B(0, j);
      block_C = &C(i, j);
      if (n == 4)
        kernel_mm256_8x4xk(8, ldb, ldc, K, Ai, block_B, block_C);
      if (n == 3)
        kernel_mm256_8x3xk(8, ldb, ldc, K, Ai, block_B, block_C);
      if (n == 2)
        kernel_mm256_8x2xk(8, ldb, ldc, K, Ai, block_B, block_C);
      else
        kernel_mm256_8x1xk(8, ldb, ldc, K, Ai, block_B, block_C);
    }
    m -= 8;
    i += 8;
  }
#endif

#define LAST_LDA 16

#ifdef DRIVER_USE_12
#undef LAST_LDA
#define LAST_LDA 12
#endif

#ifdef DRIVER_USE_8
#undef LAST_LDA
#define LAST_LDA 8
#endif

  if (m != 0)
  {
    pack(Ai, LAST_LDA, &A(i, 0), lda, m, K);
    for (int j = 0; j < N; j += 16)
    {
      int block_N = min(16, N - j);
      block_B = &B(0, j);
      block_C = &C(i, j);
      kernel_naive(LAST_LDA, ldb, ldc, m, block_N, K, Ai, block_B, block_C);
    }
  } // End dealing with remainder m x N, m < 16
}

// Level 2 blocking, k-i-j
inline void do_block_L2(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  const double *block_A, *block_B;
  double *block_C;
  for (int k = 0; k < K; k += K2)
  {
    int block_K = min(K2, K - k);
    for (int i = 0; i < M; i += M2)
    {
      int block_M = min(M2, M - i);
      for (int j = 0; j < N; j += N2)
      {
        int block_N = min(N2, N - j);
        block_A = &A(i, k);
        block_B = &B(k, j);
        block_C = &C(i, j);
        kernel_driver(lda, ldb, ldc, block_M, block_N, block_K, block_A, block_B, block_C);
      }
    }
  }
}

// Level 1 blocking, k-i-j
inline void do_block_L1(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  const double *block_A, *block_B;
  double *block_C;
  for (int k = 0; k < K; k += K1)
  {
    int block_K = min(K1, K - k);
    for (int i = 0; i < M; i += M1)
    {
      int block_M = min(M1, M - i);
      for (int j = 0; j < N; j += N1)
      {
        int block_N = min(N1, N - j);
        block_A = &A(i, k);
        block_B = &B(k, j);
        block_C = &C(i, j);
        do_block_L2(lda, ldb, ldc, block_M, block_N, block_K, block_A, block_B, block_C);
      }
    }
  }
}

void square_dgemm(int lda, const double *restrict A, const double *restrict B, double *restrict C)
{
  do_block_L1(lda, lda, lda, lda, lda, lda, A, B, C);
}
