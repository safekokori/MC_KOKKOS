#ifndef UTILS_H
#define UTILS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
// all class need to be POD
typedef float Real;
constexpr Real REALMIN  = std::numeric_limits<Real>::min();
constexpr Real REALMAX  = std::numeric_limits<Real>::max();
constexpr Real REALEPS  = std::numeric_limits<Real>::epsilon();
constexpr Real NANVALUE = std::numeric_limits<Real>::quiet_NaN();
typedef size_t Index;
constexpr Index ILLEGAL_INDEX = std::numeric_limits<Index>::quiet_NaN();
typedef struct Vec3f
{
    static struct Vec3f NAN_VEC3F()
    {
        static Vec3f st(NANVALUE, NANVALUE, NANVALUE);
        return st;
    };
    Real x, y, z;
    Vec3f(Real x, Real y, Real z) : x(x), y(y), z(z) {}
    Vec3f() {}
    bool operator==(const Vec3f &p) const
    {
        return (x - p.x) * (x - p.x) < REALEPS && (y - p.y) * (y - p.y) < REALEPS && (z - p.z) * (z - p.z) < REALEPS;
    }
    Vec3f operator+(const Vec3f &p) const { return Vec3f(x + p.x, y + p.y, z + p.z); }
    Vec3f operator-(const Vec3f &p) const { return Vec3f(x - p.x, y - p.y, z - p.z); }
    Vec3f operator*(const Real &s) const { return Vec3f(x * s, y * s, z * s); }
    Vec3f operator/(const Real &s) const { return Vec3f(x / s, y / s, z / s); }
    double dot(const Vec3f &p) const { return x * p.x + y * p.y + z * p.z; }
    Vec3f cross(const Vec3f &p) const { return Vec3f(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x); }
    double norm() const { return sqrt(x * x + y * y + z * z); }
    Vec3f normalize() { return *this / norm(); }
} Vec3f;

class RandomUtils
{
   public:
    RandomUtils(unsigned int sd) : random_pool(sd) {}

    double generate_random_double(double lower_bound, double upper_bound)
    {
        auto generator      = random_pool.get_state();
        double random_value = generator.drand(lower_bound, upper_bound);
        random_pool.free_state(generator);
        return random_value;
    }
    double generate_random_double() { return generate_random_double(0, 1); }
    int generate_random_int(int lower_bound, int upper_bound)
    {
        auto generator   = random_pool.get_state();
        int random_value = generator.urand(lower_bound, upper_bound + 1);
        random_pool.free_state(generator);
        return random_value;
    }
    int generate_random_int() { return generate_random_int(0, std::numeric_limits<int>::max()); }

   private:
    Kokkos::Random_XorShift64_Pool<> random_pool;
};
static RandomUtils &GetRandomUtils()
{
    static RandomUtils instance(time(NULL));
    return instance;
}

#endif  // UTILS_H