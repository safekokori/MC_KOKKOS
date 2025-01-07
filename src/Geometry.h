#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "Utils.h"
typedef Vec3f Point;
class Face
{
   public:
    Point p1, p2, p3;
    KOKKOS_INLINE_FUNCTION
    Face(Point p1, Point p2, Point p3) : p1(p1), p2(p2), p3(p3) {}
    KOKKOS_INLINE_FUNCTION
    Face(){}
    KOKKOS_INLINE_FUNCTION
    Point normal() const { return (p2 - p1).cross(p3 - p1).normalize(); }
    KOKKOS_INLINE_FUNCTION
    bool operator==(const Face &f) const { return p1 == f.p1 && p2 == f.p2 && p3 == f.p3; }
};

class Pyramid
{
   public:
    Point p1, p2, p3, p4;
    Face f1, f2, f3, f4;
    typedef struct _value
    {
        Scalar mua = NANVALUE, mus = NANVALUE, g = NANVALUE, n = NANVALUE;
    } Attribute;
    Attribute value;
    KOKKOS_INLINE_FUNCTION
    Pyramid(Point p1, Point p2, Point p3, Point p4)
        : p1(p1),
          p2(p2),
          p3(p3),
          p4(p4),
          f1(p1, p2, p3),
          f2(p1, p2, p4),
          f3(p1, p3, p4),
          f4(p2, p3, p4)
    {
    }
    KOKKOS_INLINE_FUNCTION
    Pyramid() : p1(), p2(), p3(), p4(), 
                f1(), f2(), f3(), f4(),
                value() {}
    KOKKOS_INLINE_FUNCTION
    bool operator==(const Pyramid &p) const
    {
        return p1 == p.p1 && p2 == p.p2 && p3 == p.p3 && p4 == p.p4;
    }
    KOKKOS_INLINE_FUNCTION
    bool HasFace(const Face &f) const { return f1 == f || f2 == f || f3 == f || f4 == f; }
    KOKKOS_INLINE_FUNCTION
    bool HasFace(const Point &p1, const Point &p2, const Point &p3) const
    {
        return f1 == Face(p1, p2, p3) || f2 == Face(p1, p2, p3) || f3 == Face(p1, p2, p3) ||
               f4 == Face(p1, p2, p3);
    }
    KOKKOS_INLINE_FUNCTION
    bool isOnSameSide(const Point &p, const Face &face) const {
        Point normal = face.normal();
        Scalar d = -(normal.dot(face.p1));
        Scalar distance = normal.dot(p) + d;
        return distance >= 0; // Assuming the normal points outwards from the pyramid
    }
    KOKKOS_INLINE_FUNCTION
    bool InPyramid(const Point &p) const {
        return isOnSameSide(p, f1) && isOnSameSide(p, f2) && isOnSameSide(p, f3) && isOnSameSide(p, f4);
    }
};
class IntersectionUtils
{
   public:
    struct IntersectionResult
    {
        Scalar t  = NANVALUE;
        Scalar b1 = NANVALUE;
        Scalar b2 = NANVALUE;

        bool hit = false;
        Vec3f ps{NANVALUE, NANVALUE, NANVALUE};
        Point p1 = ps;  // First point of the triangle or edge/vertex
        Point p2 = ps;  // Second point of the triangle or edge
        Point p3 = ps;  // Third point of the triangle or illegal
                                        // value if not applicable
        int type = 0;                   // 1: vertex, 2: edge, 3: face
    };
    // Möller–Trumbore intersection algorithm
    KOKKOS_INLINE_FUNCTION
    static IntersectionResult ray_triangle_intersection(const Point &P0, const Point &P1,
                                                        const Point &P2, const Point &orig,
                                                        const Point &dir)
    {
        IntersectionResult result;
        auto S  = orig - P0;
        auto E1 = P1 - P0;
        auto E2 = P2 - P0;
        auto S1 = dir.cross(E2);
        auto S2 = S.cross(E1);

        auto divisor = S1.dot(E1);
        if (divisor == 0)
        {
            result.hit = false;
            return result;
        }
        result.t  = S2.dot(E2) / divisor;
        result.b1 = S1.dot(S) / divisor;
        result.b2 = S2.dot(dir) / divisor;

        if (result.t < 0 || result.b1 < 0 || result.b2 < 0 || result.b1 + result.b2 > 1)
        {
            result.hit = false;
            return result;
        }

        result.hit = true;
        // Determine the location of the intersection point
        if (result.b1 > 0 && result.b1 < 1 && result.b2 > 0 && result.b2 < 1 &&
            result.b1 + result.b2 < 1)
        {
            // Intersection is inside the triangle.
            result.p1   = P0;
            result.p2   = P1;
            result.p3   = P2;
            result.type = 3;
        }
        else if ((result.b1 == 0 && result.b2 > 0 && result.b1 + result.b2 == 1))
        {
            // Intersection is on edge P0-P2.
            result.p1   = P0;
            result.p2   = P2;
            result.type = 2;
        }
        else if ((result.b1 > 0 && result.b2 == 0 && result.b1 + result.b2 == 1))
        {
            // Intersection is on edge P0-P1.
            result.p1   = P0;
            result.p2   = P1;
            result.type = 2;
        }
        else if ((result.b1 > 0 && result.b2 > 0 && result.b1 + result.b2 == 1))
        {
            // Intersection is on edge P1-P2.
            result.p1   = P1;
            result.p2   = P2;
            result.type = 2;
        }
        else if ((result.b1 == 0 && result.b2 == 0))
        {
            // Intersection is at vertex P0.
            result.p1   = P0;
            result.type = 1;
        }
        else if ((result.b1 == 1 && result.b2 == 0))
        {
            // Intersection is at vertex P1.
            result.p1   = P1;
            result.type = 1;
        }
        else if ((result.b1 == 0 && result.b2 == 1))
        {
            // Intersection is at vertex P2.
            result.p1   = P2;
            result.type = 1;
        }
        else
        {
            result.type = 0;
        }
        return result;
    }
    KOKKOS_INLINE_FUNCTION
    static IntersectionResult ray_triangle_intersection(const Face &face, const Point &orig,
                                                        const Point &dir)
    {
        return ray_triangle_intersection(face.p1, face.p2, face.p3, orig, dir);
    }
    struct IntersectionResult4{
        IntersectionResult result[4];
    };
    KOKKOS_INLINE_FUNCTION
    static IntersectionResult4 ray_pyramid_intersection(const Pyramid &pyramid,
                                                                        const Point &orig,
                                                                        const Point &dir)
    {
        IntersectionResult4 results;

        results.result[0] = ray_triangle_intersection(pyramid.f1, orig, dir);
        results.result[1] = ray_triangle_intersection(pyramid.f2, orig, dir);
        results.result[2] = ray_triangle_intersection(pyramid.f3, orig, dir);
        results.result[3] = ray_triangle_intersection(pyramid.f4, orig, dir);

        return results;
    }
};
#endif