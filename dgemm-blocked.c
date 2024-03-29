const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
    {
      double cij = C[i + j * lda];
      for (int k = 0; k < K; ++k)
        cij += A[i + k * lda] * B[k + j * lda];
      C[i + j * lda] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min(BLOCK_SIZE, lda - i);
        int N = min(BLOCK_SIZE, lda - j);
        int K = min(BLOCK_SIZE, lda - k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
      }
}
