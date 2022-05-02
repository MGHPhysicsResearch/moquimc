///\file
///\brief Error check functions for CUDA and CPU
#ifndef MQI_ERROR_CHECK_HPP
#define MQI_ERROR_CHECK_HPP

namespace mqi
{
///\ check last CUDA error
/// this is not to type __CUDACC__ directive, so use at it is in the code
inline void
check_cuda_last_error(const char* msg) {

#if defined(__CUDACC__)

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        printf(
          "current total %f MB free %f MB\n", total / (1024.0 * 1024.0), free / (1024.0 * 1024.0));
        printf("CUDA error: %s %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }

#endif
}

}   // namespace mqi

// CUDA specific
#if defined(__CUDACC__)
#define gpu_err_chk(ans)                                                                           \
    { cuda_error_checker((ans), __FILE__, __LINE__); }

inline void
cuda_error_checker(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif

#endif