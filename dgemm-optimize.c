#include <immintrin.h>

const char *dgemm_desc = "Apricity's optimized dgemm.";

#define M1 128
#define N1 128
#define K1 256

#define M2 128
#define N2 128
#define K2 32

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define A(i, k) (A[i + k * lda])
#define B(k, j) (B[k + j * ldb])
#define C(i, j) (C[i + j * ldc])

#define AVX2_LOAD_A(ymm, i, k) (ymm = _mm256_load_pd((A) + (i) + (k)*lda))
#define AVX2_LOAD_C(ymm, i, j) (ymm = _mm256_load_pd((C) + (i) + (j)*ldc))
#define AVX2_BROADCAST_B(ymm, k, j) (ymm = _mm256_broadcast_sd((B) + (k) + (j)*ldb))
#define AVX2_FMADD(c, a, b) (c = _mm256_fmadd_pd(a, b, c))
#define AVX2_STORE_C(ymm, i, j) (_mm256_store_pd((C) + (i) + (j)*ldc, ymm))

static inline void kernel_avx2_8x4x1(int lda, int ldb, int ldc, int kc, const double *restrict A, const double *restrict B, double *restrict C)
{
  __assume_aligned(A, 32);
  // __assume_aligned(BT, 32);
  // __assume_aligned(C, 32);

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

// naive kernel
static inline void kernel_naive(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  for (int j = 0; j < N; ++j)
  {
    for (int k = 0; k < K; ++k)
    {
      for (int i = 0; i < M; ++i)
        C(i, j) += A(i, k) * B(k, j);
    }
  }
}

static inline void pack(double *restrict dest, int ld_dest, const double *restrict src, int ld_src, int X, int Y)
{
  for (int x = 0; x < X; x++)
    for (int y = 0; y < Y; y++)
      dest[x + y * ld_dest] = src[x + y * ld_src];
}

static inline void packT(double *restrict dest, int ld_dest, const double *restrict src, int ld_src, int X, int Y)
{
  for (int x = 0; x < X; x++)
    for (int y = 0; y < Y; y++)
      dest[x * ld_dest + y] = src[x + y * ld_src];
}

// Register blocking, i-j, K unchanged
#define Mc 8
#define Nc 4
#define Kc K
double Ai[Mc * K2] __attribute((aligned(32)));
#define Ai(k) (Ai[(k)*Mc])
static inline void do_block_8x4(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
{
  const double *restrict block_A;
  const double *restrict block_B;
  double *restrict block_C;

  int i;
  for (i = 0; i <= M - Mc; i += Mc)
  {
    int block_M = Mc;
    // block_A = &A(i, 0);
    pack(Ai, Mc, &A(i, 0), lda, block_M, K);
    int j;
    for (j = 0; j <= N - Nc; j += 4)
    {
      block_A = &Ai(0);
      block_B = &B(0, j);
      block_C = &C(i, j);
      kernel_avx2_8x4x1(Mc, ldb, ldc, K, block_A, block_B, block_C);
    }
    // end for(j), (N-j) columns remain in C
    // block_M = 8, block_N = (N-j), block_K = K
    if (j != N)
      kernel_naive(Mc, ldb, ldc, 8, N - j, K, &Ai(0), &B(0, j), &C(i, j));
  }
  // end for(i), (M-i) rows remain in C
  // block_M = (M-i), block_N = N, block_K = K
  if (i != M)
    kernel_naive(lda, ldb, ldc, M - i, N, K, &A(i, 0), B, &C(i, 0));
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
        do_block_8x4(lda, ldb, ldc, block_M, block_N, block_K, block_A, block_B, block_C);
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
