/*************************************************************************
> File Name: Helpers.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Helper functions compatibles with CUDA.
> Created Time: 2018/01/14
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_HELPERS_H
#define SNOW_SIMULATION_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>

#define LAUNCH(...) { __VA_ARGS__; cudaDeviceSynchronize(); }

#define cudaMallocAndCopy(dst, src, size)                      \
({                                                             \
    cudaMalloc((void**)&dst, size);                            \
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);        \
})

#define TEST(toCheck, msg, failExprs)       \
({                                          \
    if (toCheck)                            \
    {                                       \
        printf("[PASSED]: %s\n", msg);      \
    }                                       \
    else                                    \
    {                                       \
        printf("[FAILED]: %s\n", msg);      \
        failExprs;                          \
    }                                       \
})

constexpr int THREAD_COUNT = 128;

#endif