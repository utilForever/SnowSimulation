/*************************************************************************
> File Name: Vector.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Vector type compatibles with CUDA.
> Created Time: 2017/12/30
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_VECTOR_H
#define SNOW_SIMULATION_VECTOR_H

#include <cuda_runtime.h>

#ifdef __CUDACC_
#include <Math.h>
#endif

#include <Common/Math.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec3.hpp>

#include <stdio.h>

struct Vec3
{
    union
    {
        float data[3];
        struct
        {
            float x;
            float y;
            float z;
        };
    };

    __host__ __device__ __forceinline__
        Vec3()
    {
        x = 0.f;
        y = 0.f;
        z = 0.f;
    }

    __host__ __device__ __forceinline__
        Vec3(float v)
    {
        x = v;
        y = v;
        z = v;
    }

    __host__ __device__ __forceinline__
        Vec3(float xx, float yy, float zz)
    {
        x = xx;
        y = yy;
        z = zz;
    }

    __host__ __device__ __forceinline__
        Vec3(const Vec3& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    __host__ __device__ __forceinline__
        Vec3(const glm::vec3& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    __host__ __device__ __forceinline__
        Vec3(const glm::ivec3& v)
    {
        x = static_cast<float>(v.x);
        y = static_cast<float>(v.y);
        z = static_cast<float>(v.z);
    }

    __host__ __device__ __forceinline__
        operator glm::vec3() const
    {
        return glm::vec3(x, y, z);
    }

    __host__ __device__ __forceinline__
        operator glm::ivec3() const
    {
        return glm::ivec3(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
    }

    __host__ __device__ __forceinline__
        Vec3& operator=(const Vec3& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3& operator=(const glm::vec3& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3& operator=(const glm::ivec3& rhs)
    {
        x = static_cast<float>(rhs.x);
        y = static_cast<float>(rhs.y);
        z = static_cast<float>(rhs.z);
        
        return *this;
    }

    __host__ __device__ __forceinline__
        int GetMajorAxis() const
    {
        return ((std::fabs(x) > std::fabs(y)) ?
            ((std::fabs(x) > std::fabs(z)) ? 0 : 2) :
            ((std::fabs(y) > std::fabs(z)) ? 1 : 2));
    }

    __host__ __device__ __forceinline__
        float& operator[](int i)
    {
        return data[i];
    }

    __host__ __device__ __forceinline__
        float operator[](int i) const
    {
        return data[i];
    }

    __host__ __device__ __forceinline__
        static float Dot(const Vec3& a, const Vec3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ __forceinline__
        static Vec3 Cross(const Vec3& a, const Vec3& b)
    {
        return Vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
    }

    __host__ __device__ __forceinline__
        static Vec3 Floor(const Vec3& v)
    {
        return Vec3(std::floor(v.x), std::floor(v.y), std::floor(v.z));
    }

    __host__ __device__ __forceinline__
        static Vec3 Ceil(const Vec3& v)
    {
        return Vec3(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z));
    }

    __host__ __device__ __forceinline__
        static Vec3 Abs(const Vec3& v)
    {
        return Vec3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
    }

    __host__ __device__ __forceinline__
        static Vec3 Round(const Vec3& v)
    {
        return Vec3(std::round(v.x), std::round(v.y), std::round(v.z));
    }

    __host__ __device__ __forceinline__
        static float Sign(const float v)
    {
        return (0 < v) - (v < 0);
    }

    __host__ __device__ __forceinline__
        static Vec3 Sign(const Vec3& v)
    {
        return Vec3(Sign(v.x), Sign(v.y), Sign(v.z));
    }

    __host__ __device__ __forceinline__
        static Vec3 Min(const Vec3& v, const Vec3& w)
    {
        return Vec3(std::min(v.x, w.x), std::min(v.y, w.y), std::min(v.z, w.z));
    }

    __host__ __device__ __forceinline__
        static Vec3 Max(const Vec3& v, const Vec3& w)
    {
        return Vec3(std::max(v.x, w.x), std::max(v.y, w.y), std::max(v.z, w.z));
    }

    __host__ __device__ __forceinline__
        static Vec3 Mix(const Vec3& v, const Vec3& w, const Vec3& a)
    {
        return Vec3(v.x * (1.f - a.x) + w.x * a.x, v.y * (1.f - a.y) + w.y * a.y, v.z * (1.f - a.z) + w.z * a.z);
    }

    __host__ __device__ __forceinline__
        static Vec3 Mix(const Vec3& v, const Vec3& w, float a)
    {
        return Vec3(v.x * (1.f - a) + w.x * a, v.y * (1.f - a) + w.y * a, v.z * (1.f - a) + w.z * a);
    }

    __host__ __device__ __forceinline__
        static float LengthSquared(const Vec3& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    __host__ __device__ __forceinline__
        static float Length(const Vec3& v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    __host__ __device__ __forceinline__
        static Vec3 Normalize(const Vec3& v)
    {
        float f = 1.f / std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return Vec3(f * v.x, f * v.y, f * v.z);
    }

    __host__ __device__ __forceinline__
        Vec3& Mul(float f)
    {
        x *= f;
        y *= f;
        z *= f;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3& Add(float f)
    {
        x += f;
        y += f;
        z += f;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3& Add(const Vec3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3& operator+=(const Vec3& rhs)
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator+(const Vec3& rhs) const
    {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    __host__ __device__ __forceinline__
        Vec3& operator-=(const Vec3& rhs)
    {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator-(const Vec3& rhs) const
    {
        return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    __host__ __device__ __forceinline__
        Vec3& operator*=(const Vec3& rhs)
    {
        x *= rhs.x;
        y *= rhs.y;
        z *= rhs.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator*(const Vec3& rhs) const
    {
        return Vec3(x * rhs.x, y * rhs.y, z * rhs.z);
    }

    __host__ __device__ __forceinline__
        Vec3& operator/=(const Vec3& rhs)
    {
        x /= rhs.x;
        y /= rhs.y;
        z /= rhs.z;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator/(const Vec3& rhs) const
    {
        return Vec3(x / rhs.x, y / rhs.y, z / rhs.z);
    }

    __host__ __device__ __forceinline__
        Vec3& operator*=(float f)
    {
        x *= f;
        y *= f;
        z *= f;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator*(float f) const
    {
        return Vec3(f * x, f * y, f * z);
    }

    __host__ __device__ __forceinline__
        Vec3& operator*=(double d)
    {
        x = static_cast<float>(x * d);
        y = static_cast<float>(y * d);
        z = static_cast<float>(z * d);
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator*(double d) const
    {
        return Vec3(static_cast<float>(x * d), static_cast<float>(y * d), static_cast<float>(z * d));
    }

    __host__ __device__ __forceinline__
        Vec3& operator/=(float f)
    {
        float fi = 1. / f;
        x *= fi;
        y *= fi;
        z *= fi;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator/(float f) const
    {
        float fi = 1.f / f;
        return Vec3(x * fi, y * fi, z * fi);
    }

    __host__ __device__ __forceinline__
        Vec3& operator+=(float f)
    {
        x += f;
        y += f;
        z += f;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator+(float f) const
    {
        return Vec3(x + f, y + f, z + f);
    }

    __host__ __device__ __forceinline__
        Vec3& operator-=(float f)
    {
        x -= f;
        y -= f;
        z -= f;
        
        return *this;
    }

    __host__ __device__ __forceinline__
        Vec3 operator-(float f) const
    {
        return Vec3(x - f, y - f, z - f);
    }

    __host__ __device__ __forceinline__
        bool IsVlid(bool* nan = nullptr) const
    {
        if (std::isnan(x) || std::isnan(y) || std::isnan(z))
        {
            if (nan)
            {
                *nan = true;
            }

            return false;
        }
        
        if (std::isinf(x) || std::isinf(y) || std::isinf(z))
        {
            if (nan)
            {
                *nan = false;
            }

            return false;
        }

        return true;
    }

    __host__ __device__ __forceinline__
        static void Print(const Vec3& v)
    {
        printf("[%10f %10f %10f]\n", v.x, v.y, v.z);
    }

    __host__ __device__ __forceinline__
        bool operator==(const Vec3& v) const
    {
        return IsEqual(x, v.x) && IsEqual(y, v.y) && IsEqual(z, v.z);
    }

    __host__ __device__ __forceinline__
        bool operator!=(const Vec3& v) const
    {
        return IsNotEqual(x, v.x) || IsNotEqual(y, v.y) || IsNotEqual(z, v.z);
    }

};

__host__ __device__ __forceinline__
Vec3 operator-(const Vec3& v)
{
    return Vec3(-v.x, -v.y, -v.z);
}

__host__ __device__ __forceinline__
Vec3 operator*(float f, const Vec3& v)
{
    return Vec3(f * v.x, f * v.y, f * v.z);
}

__host__ __device__ __forceinline__
Vec3 operator*(double f, const Vec3& v)
{
    return Vec3(static_cast<float>(f * v.x), static_cast<float>(f * v.y), static_cast<float>(f * v.z));
}

#endif