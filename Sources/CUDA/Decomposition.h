/*************************************************************************
> File Name: Decomposition.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Decomposition functions compatibles with CUDA.
> Created Time: 2018/01/13
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_DECOMPOSITION_H
#define SNOW_SIMULATION_DECOMPOSITION_H

#include <Common/Math.h>
#include <CUDA/Matrix.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_functions.h>

// FOUR_GAMMA_SQUARED = sqrt(8)+3;
constexpr double GAMMA = 5.828427124;
// cos(pi/8)
constexpr double CSTAR = 0.923879532;
// sin(p/8)
constexpr double SSTAR = 0.3826834323;

__host__ __device__
inline void JacobiConjugation(int x, int y, int z, Matrix3& S, Quaternion& qV)
{
    // eliminate off-diagonal entries Spq, Sqp
    float ch = 2.f * (S[0] - S[4]), ch2 = ch * ch;
    float sh = S[3], sh2 = sh * sh;
    bool flag = (GAMMA * sh2 < ch2);
    float w = rsqrtf(ch2 + sh2);
    ch = flag ? w * ch : CSTAR; ch2 = ch * ch;
    sh = flag ? w * sh : SSTAR; sh2 = sh * sh;

    // build rotation matrix Q
    float scale = 1.f / (ch2 + sh2);
    float a = (ch2 - sh2) * scale;
    float b = (2.f * sh * ch) * scale;
    float a2 = a * a, b2 = b * b, ab = a * b;

    // Use what we know about Q to simplify S = Q' * S * Q
    // and the re-arranging step.
    float s0 = a2 * S[0] + 2 * ab * S[1] + b2 * S[4];
    float s2 = a * S[2] + b * S[5];
    float s3 = (a2 - b2) * S[1] + ab * (S[4] - S[0]);
    float s4 = b2 * S[0] - 2 * ab * S[1] + a2 * S[4];
    float s5 = a * S[7] - b * S[6];
    float s8 = S[8];
    S = Matrix3(
        s4, s5, s3,
        s5, s8, s2,
        s3, s2, s0);

    Vector3 tmp(sh * qV.x, sh * qV.y, sh * qV.z);
    sh *= qV.w;
    // original
    qV *= ch;

    qV[z] += sh;
    qV.w -= tmp[z];
    qV[x] += tmp[y];
    qV[y] -= tmp[x];
}

// Wrapper function for the first step. Solve symmetric
// eigen problem using Jacobi iteration. Given a symmetric
// matrix S, diagonalize it also returns the cumulative
// rotation as a Quaternion.
__host__ __device__ __forceinline__
void JacobiEigenanalysis(Matrix3& S, Quaternion& qV)
{
    qV = Quaternion(1, 0, 0, 0);

    JacobiConjugation(0, 1, 2, S, qV);
    JacobiConjugation(1, 2, 0, S, qV);
    JacobiConjugation(2, 0, 1, S, qV);

    JacobiConjugation(0, 1, 2, S, qV);
    JacobiConjugation(1, 2, 0, S, qV);
    JacobiConjugation(2, 0, 1, S, qV);

    JacobiConjugation(0, 1, 2, S, qV);
    JacobiConjugation(1, 2, 0, S, qV);
    JacobiConjugation(2, 0, 1, S, qV);

    JacobiConjugation(0, 1, 2, S, qV);
    JacobiConjugation(1, 2, 0, S, qV);
    JacobiConjugation(2, 0, 1, S, qV);
}

inline void CondSwap(bool condition, float& x, float& y)
{
    float temp = x;
    x = condition ? y : x;
    y = condition ? temp : y;
}

inline void CondNegativeSwap(bool condition, Vector3& x, Vector3& y)
{
    Vector3 temp = -x;
    x = condition ? y : x;
    y = condition ? temp : y;
}

__host__ __device__ __forceinline__
void SortSingularValues(Matrix3& B, Matrix3& V)
{
    // used in step 2
    Vector3 b1 = B.GetColumn(0); 
    Vector3 b2 = B.GetColumn(1);
    Vector3 b3 = B.GetColumn(2);
    Vector3 v1 = V.GetColumn(0);
    Vector3 v2 = V.GetColumn(1);
    Vector3 v3 = V.GetColumn(2);
    float rho1 = Vector3::Dot(b1, b1);
    float rho2 = Vector3::Dot(b2, b2);
    float rho3 = Vector3::Dot(b3, b3);
    bool c;

    c = rho1 < rho2;
    CondNegativeSwap(c, b1, b2);
    CondNegativeSwap(c, v1, v2);
    CondSwap(c, rho1, rho2);

    c = rho1 < rho3;
    CondNegativeSwap(c, b1, b3);
    CondNegativeSwap(c, v1, v3);
    CondSwap(c, rho1, rho3);

    c = rho2 < rho3;
    CondNegativeSwap(c, b2, b3);
    CondNegativeSwap(c, v2, v3);

    // rebuild B,V
    B = Matrix3(b1, b2, b3);
    V = Matrix3(v1, v2, v3);
}

__host__ __device__ __forceinline__
void QRGivensQuaternion(float a1, float a2, float& ch, float& sh)
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    float rho = sqrtf(a1 * a1 + a2 * a2);

    sh = rho > std::numeric_limits<float>::epsilon() ? a2 : 0;
    ch = fabsf(a1) + fmaxf(rho, std::numeric_limits<float>::epsilon());
    bool b = a1 < 0;

    CondSwap(b, sh, ch);

    float w = rsqrtf(ch * ch + sh * sh);

    ch *= w;
    sh *= w;
}

__host__ __device__
inline void QRDecomposition(const Matrix3& B, Matrix3& Q, Matrix3& R)
{
    R = B;

    // QR decomposition of 3x3 matrices using Givens rotations to
    // eliminate elements B21, B31, B32
    Quaternion qQ;
    Matrix3 U;
    float ch, sh, s0, s1;

    // first givens rotation
    QRGivensQuaternion(R[0], R[1], ch, sh);

    s0 = 1 - 2 * sh * sh;
    s1 = 2 * sh * ch;
    U = Matrix3(
        s0, s1, 0,
        -s1, s0, 0,
        0, 0, 1);

    R = Matrix3::MultiplyATB(U, R);

    // update cumulative rotation
    qQ = Quaternion(ch * qQ.w - sh * qQ.z, ch * qQ.x + sh * qQ.y, ch * qQ.y - sh * qQ.x, sh * qQ.w + ch * qQ.z);

    // second givens rotation
    QRGivensQuaternion(R[0], R[2], ch, sh);

    s0 = 1 - 2 * sh * sh;
    s1 = 2 * sh * ch;
    U = Matrix3(
        s0, 0, s1,
        0, 1, 0,
        -s1, 0, s0);

    R = Matrix3::MultiplyATB(U, R);

    // update cumulative rotation
    qQ = Quaternion(ch * qQ.w + sh * qQ.y, ch * qQ.x + sh * qQ.z, ch * qQ.y - sh * qQ.w, ch * qQ.z - sh * qQ.x);

    // third Givens rotation
    QRGivensQuaternion(R[4], R[5], ch, sh);

    s0 = 1 - 2 * sh * sh;
    s1 = 2 * sh * ch;
    U = Matrix3(
        1, 0, 0,
        0, s0, s1,
        0, -s1, s0);

    R = Matrix3::MultiplyATB(U, R);

    // update cumulative rotation
    qQ = Quaternion(ch * qQ.w - sh * qQ.x, sh * qQ.w + ch * qQ.x, ch * qQ.y + sh * qQ.z, ch * qQ.z - sh * qQ.y);

    // qQ now contains final rotation for Q
    Q = Matrix3::FromQuaternion(qQ);
}

// McAdams, Selle, Tamstorf, Teran, and Sifakis. Computing the Singular Value Decomposition of 3 x 3
// matrices with minimal branching and elementary floating point operations
// Computes SVD of 3x3 matrix A = W * S * V'
__host__ __device__ __forceinline__
void ComputeSVD(const Matrix3& A, Matrix3& W, Matrix3& S, Matrix3& V)
{
    // 1. Normal Quaternions matrix
    Matrix3 ATA = Matrix3::MultiplyATB(A, A);

    // 2. Symmetric Eigen analysis
    Quaternion qV;
    JacobiEigenanalysis(ATA, qV);
    V = Matrix3::FromQuaternion(qV);
    Matrix3 B = A * V;

    // 3. Sorting the singular values (find V)
    SortSingularValues(B, V);

    // 4. QR decomposition
    QRDecomposition(B, W, S);
}

// Returns polar decomposition of 3x3 matrix M where
// M = Fe = Re * Se = U * P
// U is an orthonormal matrix
// S is symmetric positive semidefinite
// Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
__host__ __device__
inline void ComputePD(const Matrix3& A, Matrix3& R)
{
    // U is unitary matrix (i.e. orthogonal/orthonormal)
    // P is positive semidefinite Hermitian matrix
    Matrix3 W, S, V;
    ComputeSVD(A, W, S, V);
    R = Matrix3::MultiplyABT(W, V);
}

// Returns polar decomposition of 3x3 matrix M where
// M = Fe = Re * Se = U * P
// U is an orthonormal matrix
// S is symmetric positive semidefinite
// Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
__host__ __device__
inline void ComputePD(const Matrix3& A, Matrix3& R, Matrix3& P)
{
    // U is unitary matrix (i.e. orthogonal/orthonormal)
    // P is positive semidefinite Hermitian matrix
    Matrix3 W, S, V;
    
    ComputeSVD(A, W, S, V);
    
    R = Matrix3::MultiplyABT(W, V);
    P = Matrix3::MultiplyADBT(V, S, V);
}

// In snow we desire both SVD and polar decompositions simultaneously without
// re-computing USV for polar.
// here is a function that returns all the relevant values
// SVD : A = W * S * V'
// PD : A = R * E
__host__ __device__
inline void ComputeSVDandPD(const Matrix3& A, Matrix3& W, Matrix3& S, Matrix3& V, Matrix3& R)
{
    ComputeSVD(A, W, S, V);

    R = Matrix3::MultiplyABT(W, V);
}

#endif