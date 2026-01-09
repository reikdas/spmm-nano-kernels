//
// Wrapper implementation for sparse-register-tiling SpMM
//

#include "spmm_spreg_wrapper.h"

#include <cstring>
#include <iostream>

#include "MatMulSpecialized.h"
#include "KernelDesc.h"
#include "Config.h"
#include "mapping_io.h"
#include "mapping_to_executor.h"

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
            true  // allow_row_padding
        );
        
        // Allocate the executor for the given N (bcols)
        matmul->allocate_executor(N);
        
        // Create and return handle
        SpregHandle* handle = new SpregHandle();
        handle->matmul = matmul;
        handle->M = M;
        handle->K = K;
        handle->N = N;
        
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
    
    // Zero the output matrix
    std::memset(C, 0, h->M * h->N * sizeof(double));
    
    // Execute SpMM: C = A * B
    (*h->matmul)(C, B);
}

void spmm_spreg_cleanup(void* handle) {
    if (!handle) return;
    
    SpregHandle* h = static_cast<SpregHandle*>(handle);
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
