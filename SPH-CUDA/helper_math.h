#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#include "src/MarchingCubes/cutil_math.h"

// Norm
inline __host__ __device__ float norm(float3 v)
{
    return sqrtf(dot(v, v));
}

// Floor
inline __host__ __device__ int3 floor(float3 v)
{
    int3 floored_num = make_int3(int(v.x - (v.x - floorf(v.x))), int(v.y - (v.y - floorf(v.y))), int(v.z - (v.z - floorf(v.z))));
    return floored_num;
}

// Ceil
inline __host__ __device__ int3 ceil(float3 v)
{
    int3 ceiled_num = make_int3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
    return ceiled_num;
}

// Pow
inline __host__ __device__  float3 pow(float3 x, float y)
{
    return make_float3(expf(x.x * logf(y)), expf(x.y * logf(y)), expf(x.z * logf(y)));
}

// stream operator
inline __host__ std::ostream& operator<<(std::ostream& stream, const float3& v)
{
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return stream;
}