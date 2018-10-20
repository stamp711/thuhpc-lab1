#include "helper.h"
#include "kernel/naive.h"
#include "kernel/mm256_8x4xk.h"
#include "kernel/mm256_16x3xk.h"
#include "kernel/mm256_16x2xk.h"
#include "kernel/mm256_16x1xk.h"

const char *dgemm_desc = "Apricity's optimized dgemm.";

// best
// #define M1 64
// #define N1 120
// #define K1 256

// #define M2 64
// #define N2 120
// #define K2 32

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
static inline void do_block_kernel(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
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
    } // End deal with remadinder 16 x n
  }   // End main blocking for M

  // Calculate block remainder for M
  int m = M % Mc;

  // Deal with remainder m x N, m < 16
  if (m != 0)
    kernel_naive(lda, ldb, ldc, m, N, K, &A(i, 0), B, &C(i, 0));
}

// Level 2 blocking, k-i-j
// double Bk[K1 * N1] __attribute((aligned(32)));
// #define Bk(j) (Bk[(j)*K2])
static inline void do_block_L2(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  const double *block_A, *block_B;
  double *block_C;

  for (int k = 0; k < K; k += K2)
  {
    int block_K = min(K2, K - k);
    // pack(Bk, K2, &B(k, 0), ldb, block_K, N);
    for (int i = 0; i < M; i += M2)
    {
      int block_M = min(M2, M - i);
      for (int j = 0; j < N; j += N2)
      {
        int block_N = min(N2, N - j);
        block_A = &A(i, k);
        block_B = &B(k, j);
        block_C = &C(i, j);
        do_block_kernel(lda, ldb, ldc, block_M, block_N, block_K, block_A, block_B, block_C);
      }
    }
  }
}

// Level 1 blocking, k-i-j
static inline void do_block_L1(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
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
