#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "Utils.h"
typedef Vec3f Point;
class Face
{
   public:
    Point p1, p2, p3;
    Face(Point p1, Point p2, Point p3) : p1(p1), p2(p2), p3(p3) {}
    Point normal() const { return (p2 - p1).cross(p3 - p1).normalize(); }
    bool operator==(const Face &f) const { return p1 == f.p1 && p2 == f.p2 && p3 == f.p3; }
};

class Pyramid
{
   public:
    Point p1, p2, p3, p4;
    Face f1, f2, f3, f4;
    typedef struct _value
    {
        Real mua = NANVALUE, mus = NANVALUE, g = NANVALUE, n = NANVALUE;
    } Attribute;
    Attribute value;
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
    bool operator==(const Pyramid &p) const
    {
        return p1 == p.p1 && p2 == p.p2 && p3 == p.p3 && p4 == p.p4;
    }
    bool HasFace(const Face &f) const { return f1 == f || f2 == f || f3 == f || f4 == f; }
    bool HasFace(const Point &p1, const Point &p2, const Point &p3) const
    {
        return f1 == Face(p1, p2, p3) || f2 == Face(p1, p2, p3) || f3 == Face(p1, p2, p3) ||
               f4 == Face(p1, p2, p3);
    }
    bool isOnSameSide(const Point &p, const Face &face) const {
        Point normal = face.normal();
        Real d = -(normal.dot(face.p1));
        Real distance = normal.dot(p) + d;
        return distance >= 0; // Assuming the normal points outwards from the pyramid
    }
    bool InPyramid(const Point &p) const {
        return isOnSameSide(p, f1) && isOnSameSide(p, f2) && isOnSameSide(p, f3) && isOnSameSide(p, f4);
    }
};
class IntersectionUtils
{
   public:
    struct IntersectionResult
    {
        Real t  = NANVALUE;
        Real b1 = NANVALUE;
        Real b2 = NANVALUE;

        bool hit = false;

        Point p1 = Point::NAN_VEC3F();  // First point of the triangle or edge/vertex
        Point p2 = Point::NAN_VEC3F();  // Second point of the triangle or edge
        Point p3 = Point::NAN_VEC3F();  // Third point of the triangle or illegal
                                        // value if not applicable
        int type = 0;                   // 1: vertex, 2: edge, 3: face
    };
    // Möller–Trumbore intersection algorithm
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
    static IntersectionResult ray_triangle_intersection(const Face &face, const Point &orig,
                                                        const Point &dir)
    {
        return ray_triangle_intersection(face.p1, face.p2, face.p3, orig, dir);
    }
    static Kokkos::View<IntersectionResult[4]> ray_pyramid_intersection(const Pyramid &pyramid,
                                                                        const Point &orig,
                                                                        const Point &dir)
    {
        Kokkos::View<IntersectionResult[4]> results("IntersectionResult");

        results(0) = ray_triangle_intersection(pyramid.f1, orig, dir);
        results(1) = ray_triangle_intersection(pyramid.f2, orig, dir);
        results(2) = ray_triangle_intersection(pyramid.f3, orig, dir);
        results(3) = ray_triangle_intersection(pyramid.f4, orig, dir);

        return results;
    }
};
#endif