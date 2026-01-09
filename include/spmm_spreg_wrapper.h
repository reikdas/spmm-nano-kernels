//
// Wrapper for sparse-register-tiling SpMM functionality
// Provides a simple C-compatible interface for SABLE codegen
//

#ifndef SPMM_SPREG_WRAPPER_H
#define SPMM_SPREG_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the sparse-register-tiling SpMM executor.
 * This performs inspection and packing of the sparse matrix.
 * Call this ONCE outside the timing loop.
 *
 * @param csr_val     CSR values array (double*)
 * @param indices     CSR column indices array (int*)
 * @param indptr      CSR row pointers array (int*)
 * @param M           Number of rows in sparse matrix
 * @param K           Number of columns in sparse matrix
 * @param N           Number of columns in dense matrix B (e.g., 512)
 * @return            Opaque handle to the initialized executor (void*)
 */
void* spmm_spreg_init(
    const double* csr_val,
    const int* indices,
    const int* indptr,
    int M,
    int K,
    int N
);

/**
 * Execute SpMM: C = A * B
 * Call this inside the timing loop.
 *
 * @param handle      Opaque handle from spmm_spreg_init
 * @param C           Output matrix (M x N), row-major
 * @param B           Dense input matrix (K x N), row-major
 */
void spmm_spreg_execute(
    void* handle,
    double* C,
    const double* B
);

/**
 * Cleanup and free the executor.
 * Call this after all executions are done.
 *
 * @param handle      Opaque handle from spmm_spreg_init
 */
void spmm_spreg_cleanup(void* handle);

/**
 * Convenience function that combines init + execute + cleanup.
 * Use this for one-shot SpMM (not recommended for benchmarking).
 *
 * @param C           Output matrix (M x N), row-major
 * @param csr_val     CSR values array (double*)
 * @param indices     CSR column indices array (int*)
 * @param indptr      CSR row pointers array (int*)
 * @param B           Dense input matrix (K x N), row-major
 * @param M           Number of rows in sparse matrix
 * @param K           Number of columns in sparse matrix
 * @param N           Number of columns in dense matrix B
 */
void spmm_spreg(
    double* C,
    const double* csr_val,
    const int* indices,
    const int* indptr,
    const double* B,
    int M,
    int K,
    int N
);

#ifdef __cplusplus
}
#endif

#endif /* SPMM_SPREG_WRAPPER_H */
