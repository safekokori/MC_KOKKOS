#ifndef UTILS_H
#define UTILS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

using ExecSpace    = MemSpace::execution_space;
using range_policy = Kokkos::RangePolicy<ExecSpace>;
// 添加这一行定义 RandPool 类型
typedef Kokkos::Random_XorShift64_Pool<> RandPool;

// all class need to be POD
typedef float Scalar;
constexpr Scalar REALMIN  = FLT_MIN;
constexpr Scalar REALMAX  = FLT_MAX;
constexpr Scalar REALEPS  = 1e-15;
constexpr Scalar NANVALUE = REALMIN;
KOKKOS_INLINE_FUNCTION
bool IsNan(Scalar x){
    return x <= NANVALUE+REALEPS || x >= NANVALUE-2;
}
typedef int Index;
constexpr Index ILLEGAL_INDEX = std::numeric_limits<Index>::quiet_NaN();
typedef struct Vec3f
{
    Scalar x=0, y=0, z=0;
    // KOKKOS_INLINE_FUNCTION
    // Vec3f(Scalar x, Scalar y, Scalar z) : x(x), y(y), z(z) {}
    // KOKKOS_INLINE_FUNCTION
    // Vec3f() : x(0), y(0), z(0) {}
    KOKKOS_INLINE_FUNCTION
    bool operator==(const Vec3f &p) const
    {
        return (x - p.x) * (x - p.x) < REALEPS && (y - p.y) * (y - p.y) < REALEPS && (z - p.z) * (z - p.z) < REALEPS;
    }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator+(const Vec3f &p) const { 
        Vec3f res;
        res.x = x + p.x;
        res.y = y + p.y;
        res.z = z + p.z;
        return res;
         }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator-(const Vec3f &p) const { 
        Vec3f res;
        res.x = x - p.x;
        res.y = y - p.y;
        res.z = z - p.z;
        return res;
         }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator*(const Scalar &s) const { 
        Vec3f res;
        res.x = x * s;
        res.y = y * s;
        res.z = z * s;
        return res;
         }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator/(const Scalar &s) const { 
        Vec3f res;
        res.x = x / s;
        res.y = y / s;
        res.z = z / s;
        return res;
         }
    KOKKOS_INLINE_FUNCTION
    double dot(const Vec3f &p) const { return x * p.x + y * p.y + z * p.z; }
    KOKKOS_INLINE_FUNCTION
        Vec3f cross(const Vec3f &p) const { 
        Vec3f res;
        res.x = y * p.z - z * p.y;
        res.y = z * p.x - x * p.z;
        res.z = x * p.y - y * p.x;
        return res;
         }
    KOKKOS_INLINE_FUNCTION
    double norm() const { return sqrt(x * x + y * y + z * z); }
    KOKKOS_INLINE_FUNCTION
    Vec3f normalize() { return *this / norm(); }
} Vec3f;

typedef typename Kokkos::Random_XorShift64_Pool<ExecSpace> RandPoolType;
constexpr unsigned int MAX_ITER = 100;
#endif  // UTILS_H
