#pragma once
#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

// naive kernel
inline void kernel_naive(int lda, int ldb, int ldc, int M, int N, int K, const double *restrict A, const double *restrict B, double *restrict C)
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
