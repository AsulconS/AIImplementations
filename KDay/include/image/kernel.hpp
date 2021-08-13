#if !defined(KERNEL_HPP)
#define KERNEL_HPP

#if defined(__CMAKE_SHARED_MODE)
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

#include <string>
#include <fstream>

template <int M, int N>
struct DLL_EXPORT Kernel
{
    float factor;
    int matrix[M][N];
};

template <int M, int N>
void DLL_EXPORT kernelFromFile(Kernel<M, N>& kernel, const char* name);

#include <image/kernel.inl>

#endif // KERNEL_HPP
