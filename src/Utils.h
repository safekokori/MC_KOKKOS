#ifndef UTILS_H
#define UTILS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// 添加这一行定义 RandPool 类型
typedef Kokkos::Random_XorShift64_Pool<> RandPool;

// all class need to be POD
typedef float Scalar;
constexpr Scalar REALMIN  = std::numeric_limits<Scalar>::min();
constexpr Scalar REALMAX  = std::numeric_limits<Scalar>::max();
constexpr Scalar REALEPS  = std::numeric_limits<Scalar>::epsilon();
constexpr Scalar NANVALUE = std::numeric_limits<Scalar>::quiet_NaN();
typedef size_t Index;
constexpr Index ILLEGAL_INDEX = std::numeric_limits<Index>::quiet_NaN();
typedef struct Vec3f
{
    Scalar x, y, z;
    KOKKOS_INLINE_FUNCTION
    Vec3f(Scalar x, Scalar y, Scalar z) : x(x), y(y), z(z) {}
    KOKKOS_INLINE_FUNCTION
    Vec3f() : x(0), y(0), z(0) {}
    KOKKOS_INLINE_FUNCTION
    bool operator==(const Vec3f &p) const
    {
        return (x - p.x) * (x - p.x) < REALEPS && (y - p.y) * (y - p.y) < REALEPS && (z - p.z) * (z - p.z) < REALEPS;
    }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator+(const Vec3f &p) const { return Vec3f(x + p.x, y + p.y, z + p.z); }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator-(const Vec3f &p) const { return Vec3f(x - p.x, y - p.y, z - p.z); }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator*(const Scalar &s) const { return Vec3f(x * s, y * s, z * s); }
    KOKKOS_INLINE_FUNCTION
    Vec3f operator/(const Scalar &s) const { return Vec3f(x / s, y / s, z / s); }
    KOKKOS_INLINE_FUNCTION
    double dot(const Vec3f &p) const { return x * p.x + y * p.y + z * p.z; }
    KOKKOS_INLINE_FUNCTION
    Vec3f cross(const Vec3f &p) const { return Vec3f(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x); }
    KOKKOS_INLINE_FUNCTION
    double norm() const { return sqrt(x * x + y * y + z * z); }
    KOKKOS_INLINE_FUNCTION
    Vec3f normalize() { return *this / norm(); }
} Vec3f;

#define NAN_VEC3F (Vec3f(NANVALUE, NANVALUE, NANVALUE))

typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;

#endif  // UTILS_H
