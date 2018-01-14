/*************************************************************************
> File Name: Math.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: The common math header file of snow simulation.
> Created Time: 2017/12/29
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MATH_H
#define SNOW_SIMULATION_MATH_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

template <typename T>
bool IsEqual(T a, T b)
{ 
    return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
}

template <typename T>
bool IsNotEqual(T a, T b)
{
    return !IsEqual(a, b);
}

static float UniformRandom(float min = 0.f, float max = 1.f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    return dis(gen);
}

// TODO: Delete this code after CUDA 9 supports VS 15.5
__device__
inline float Clamp(float value, float a, float b)
{
    return (value < a) ? a : ((value > b) ? b : value);
}

__device__
static float SmoothStep(float value, float edge0, float edge1)
{
    float x = Clamp((value - edge0) / (edge1 - edge0), 0.f, 1.f);
    return x * x * (3 - 2 * x);
}

__device__
static float SmootherStep(float value, float edge0, float edge1)
{
    float x = Clamp((value - edge0) / (edge1 - edge0), 0.f, 1.f);
    return x * x * x * (x * (6 * x - 15) + 10);
}

// TODO: Uncomment this code after CUDA 9 support VS 15.5
//static float SmoothStep(float value, float edge0, float edge1)
//{
//    float x = std::clamp((value - edge0) / (edge1 - edge0), 0.f, 1.f);
//    return x * x * (3 - 2 * x);
//}
//
//static float SmootherStep(float value, float edge0, float edge1)
//{
//    float x = std::clamp((value - edge0) / (edge1 - edge0), 0.f, 1.f);
//    return x * x * x * (x * (6 * x - 15) + 10);
//}

#endif