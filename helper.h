#pragma once

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define A(i, k) (A[(i) + (k)*lda])
#define B(k, j) (B[(k) + (j)*ldb])
#define C(i, j) (C[(i) + (j)*ldc])

inline void pack(double *restrict dest, int ld_dest, const double *restrict src, int ld_src, int X, int Y)
{
  for (int x = 0; x < X; x++)
    for (int y = 0; y < Y; y++)
      dest[x + y * ld_dest] = src[x + y * ld_src];
}

inline void packT(double *restrict dest, int ld_dest, const double *restrict src, int ld_src, int X, int Y)
{
  for (int x = 0; x < X; x++)
    for (int y = 0; y < Y; y++)
      dest[x * ld_dest + y] = src[x + y * ld_src];
}
