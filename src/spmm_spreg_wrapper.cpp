//
// Wrapper implementation for sparse-register-tiling SpMM
//

#include "spmm_spreg_wrapper.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "MatMulSpecialized.h"
#include "KernelDesc.h"
#include "Config.h"
#include "mapping_io.h"
#include "mapping_to_executor.h"

// M_r for double precision AVX512 4x6 tile is 4
static const int M_R = 4;

// Use Intel kernel descriptor with load balancing for double precision
// DataTransform=true means values are packed/transformed during inspection
using KernelDescType = sop::KD_IntelLoadBalanced<double>;
using MatMulType = sop::MatMulSpecialized<KernelDescType, true>;

// Wrapper struct to hold the executor and dimensions
struct SpregHandle {
    MatMulType* matmul;
    int M;
    int K;
    int N;
    int M_padded;      // M rounded up to multiple of M_r (4)
    double* internal_C; // Internal buffer for padded output (allocated if M != M_padded)
};

// Default configuration for single-threaded execution
// Based on heuristics from sparse-register-tiling for bcols=512
// Using NANO_M4N4_NKM_LB_TLB128_SA_identity as a good default
static const char* DEFAULT_MAPPING_ID = "61fee";  // identity mapping for M4

extern "C" {

void* spmm_spreg_init(
    const double* csr_val,
    const int* indices,
    const int* indptr,
    int M,
    int K,
    int N
) {
    // Create tile configuration
    // Based on bench.py defaults: bcols=512
    // Using CAKE_TILING_WITH_TLB_COMPENSATION (tiling_strategy=2)
    sop::TileConfig config;
    config.M_c = 64;        // Default M tile
    config.K_c = 256;       // Default K tile  
    config.N_c = 64;        // Default N tile
    config.tiling_strategy = sop::CAKE_TILING_WITH_TLB_COMPENSATION;
    config.max_tlb_entries = 128;
    config.tlb_page_size = 4096;
    config.sparse_a = 1;
    config.beta = 1.0f;
    config.runtimeSchedule = sop::nmNKM;  // NKM schedule
    
    // Number of threads = 1 (single-threaded)
    int num_threads = 1;
    
    // Mapping ID from heuristics (identity mapping for M4)
    std::string mapping_id = DEFAULT_MAPPING_ID;
    // Use the generated get_executor_id function with AVX512, 512-bit vectors, auto-select N_r (-1)
    std::string executor_id = get_executor_id(mapping_id, "AVX512", 512, -1);
    
    try {
        // Create the MatMulSpecialized instance
        // This performs inspection and packing
        MatMulType* matmul = new MatMulType(
            M, K, N,
            csr_val,
            indptr,
            indices,
            config,
            num_threads,
            executor_id,
            mapping_id,
            true  // allow_row_padding - must be true for CAKE tiling, caller must provide padded C
        );
        
        // Allocate the executor for the given N (bcols)
        matmul->allocate_executor(N);
        
        // Compute padded M to match library's pad_to_multiple_of behavior
        // The COO::pad_to_multiple_of always adds (M_R - M % M_R) rows,
        // even when M is already a multiple of M_R. We must match this.
        int M_padded = M + (M_R - M % M_R);
        
        // Create and return handle
        SpregHandle* handle = new SpregHandle();
        handle->matmul = matmul;
        handle->M = M;
        handle->K = K;
        handle->N = N;
        handle->M_padded = M_padded;
        
        // Allocate internal buffer only if padding is needed
        if (M_padded != M) {
            handle->internal_C = static_cast<double*>(std::aligned_alloc(64, M_padded * N * sizeof(double)));
            if (!handle->internal_C) {
                std::cerr << "Failed to allocate internal_C buffer" << std::endl;
                delete matmul;
                delete handle;
                return nullptr;
            }
        } else {
            handle->internal_C = nullptr;
        }
        
        return static_cast<void*>(handle);
        
    } catch (const std::exception& e) {
        std::cerr << "spmm_spreg_init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void spmm_spreg_execute(
    void* handle,
    double* C,
    const double* B
) {
    if (!handle) {
        std::cerr << "spmm_spreg_execute: null handle" << std::endl;
        return;
    }
    
    SpregHandle* h = static_cast<SpregHandle*>(handle);
    
    // Use internal buffer if padding is needed, otherwise use caller's buffer directly
    double* C_exec = h->internal_C ? h->internal_C : C;
    
    // Zero the output matrix (padded size if using internal buffer)
    int rows_to_zero = h->internal_C ? h->M_padded : h->M;
    std::memset(C_exec, 0, rows_to_zero * h->N * sizeof(double));
    
    // Execute SpMM: C = A * B
    (*h->matmul)(C_exec, B);
    
    // If using internal buffer, copy valid rows back to caller's buffer
    if (h->internal_C) {
        // Copy row by row since internal_C has different row stride
        std::memcpy(C, h->internal_C, h->M * h->N * sizeof(double));
    }
}

void spmm_spreg_cleanup(void* handle) {
    if (!handle) return;
    
    SpregHandle* h = static_cast<SpregHandle*>(handle);
    if (h->internal_C) {
        std::free(h->internal_C);
    }
    delete h->matmul;
    delete h;
}

void spmm_spreg(
    double* C,
    const double* csr_val,
    const int* indices,
    const int* indptr,
    const double* B,
    int M,
    int K,
    int N
) {
    void* handle = spmm_spreg_init(csr_val, indices, indptr, M, K, N);
    if (handle) {
        spmm_spreg_execute(handle, C, B);
        spmm_spreg_cleanup(handle);
    }
}

} // extern "C"
