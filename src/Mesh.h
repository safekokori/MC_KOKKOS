#ifndef MESH_H
#define MESH_H
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <fstream>
#include "Utils.h"
#include "Geometry.h"
constexpr unsigned int MAX_NEIGHBOR_COUNT   = 12;
constexpr unsigned int MAX_NEIGHBOR_COUNT_2 = MAX_NEIGHBOR_COUNT * 2;
constexpr unsigned int MAX_NEIGHBOR_COUNT_4 = MAX_NEIGHBOR_COUNT * 4;
class TetMesh
{
   private:
    bool hasMinLength = false;
    Scalar minLength    = REALMAX;

   public:
    Kokkos::View<Pyramid *, ExecSpace> pyramids;
    typedef struct Neighbor
    {
        // must be trivially copyable
        int adjacentPyramids_1[MAX_NEIGHBOR_COUNT_4];
        int adjacentPyramids_2[MAX_NEIGHBOR_COUNT_2];
        int adjacentPyramids_3[MAX_NEIGHBOR_COUNT];
        int adjacentCount_1;
        int adjacentCount_2;
        int adjacentCount_3;
    } Neighbor;
    Kokkos::UnorderedMap<int, Neighbor, ExecSpace> adjacentPyramids;
    static Scalar Distance(const Point &a, const Point &b)
    {
        Scalar dx = a.x - b.x;
        Scalar dy = a.y - b.y;
        Scalar dz = a.z - b.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    KOKKOS_FUNCTION
    Scalar GetMinLength() const {
        return hasMinLength ? minLength : 1e6 * REALEPS;
    }
    
    void requireMinLength()
    {
        auto policy = Kokkos::RangePolicy<>(0, pyramids.extent(0));
        Kokkos::parallel_reduce(
            "ComputeMinLength", policy,
            KOKKOS_LAMBDA(const int i, Scalar &localMinLength)
            {
                auto Distance = [](const Point &a, const Point &b)
                {
                    Scalar dx = a.x - b.x;
                    Scalar dy = a.y - b.y;
                    Scalar dz = a.z - b.z;
                    return std::sqrt(dx * dx + dy * dy + dz * dz);
                };
                Scalar lengths[6];
                lengths[0] = Distance(pyramids(i).p1, pyramids(i).p2);
                lengths[1] = Distance(pyramids(i).p1, pyramids(i).p3);
                lengths[2] = Distance(pyramids(i).p1, pyramids(i).p4);
                lengths[3] = Distance(pyramids(i).p2, pyramids(i).p3);
                lengths[4] = Distance(pyramids(i).p2, pyramids(i).p4);
                lengths[5] = Distance(pyramids(i).p3, pyramids(i).p4);
                for (int j = 0; j < 6; ++j)
                {
                    if (lengths[j] < localMinLength)
                    {
                        localMinLength = lengths[j];
                    }
                }
            },
            Kokkos::Min<Scalar>(minLength));
        hasMinLength = true;
    }
    void load_from_file(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("无法打开文件: " + filename);
        }

        std::string line;
        int numPoints, numTets;

        // 读取点数量
        while (std::getline(file, line))
        {
            if (line.find("points") != std::string::npos)
            {
                file >> numPoints;
                break;
            }
        }

        // 分配点数组空间
        auto points_host = Kokkos::View<Point *, Kokkos::HostSpace>("points", numPoints);

        // 读取点坐标
        for (int i = 0; i < numPoints; i++)
        {
            Scalar x, y, z;
            file >> x >> y >> z;
            points_host(i) = Point{x, y, z};
        }

        // 读取四面体数量
        while (std::getline(file, line))
        {
            if (line.find("volumeelements") != std::string::npos)
            {
                file >> numTets;
                break;
            }
        }

        // 分配四面体数组空间
        auto pyramids_host = Kokkos::View<Pyramid *, Kokkos::HostSpace>("pyramidsHost", numTets);

        // 读取四面体顶点索引
        for (int i = 0; i < numTets; i++)
        {
            int type, material, p1, p2, p3, p4;
            file >> type >> material >> p1 >> p2 >> p3 >> p4;
            // NETGEN的索引从1开始,需要减1
            pyramids_host(i) =
                Pyramid(points_host(p1 - 1), points_host(p2 - 1), points_host(p3 - 1), points_host(p4 - 1));
        }

        // 将数据从host拷贝到device
        pyramids = Kokkos::create_mirror_view_and_copy(ExecSpace(), pyramids_host);

        file.close();
        hasMinLength = false;
    }
    void buildNeighbors()
    {
        auto rangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {pyramids.extent(0), pyramids.extent(0)});
        Kokkos::parallel_for(
            "BuildNeighbors", rangePolicy,
            KOKKOS_LAMBDA(const int i, const int j)
            {
                if (i == j) return;
                int count = 0;
                if (pyramids(i).p1 == pyramids(j).p1 || pyramids(i).p1 == pyramids(j).p2 ||
                    pyramids(i).p1 == pyramids(j).p3 || pyramids(i).p1 == pyramids(j).p4)
                {
                    count++;
                }
                if (pyramids(i).p2 == pyramids(j).p1 || pyramids(i).p2 == pyramids(j).p2 ||
                    pyramids(i).p2 == pyramids(j).p3 || pyramids(i).p2 == pyramids(j).p4)
                {
                    count++;
                }
                if (pyramids(i).p3 == pyramids(j).p1 || pyramids(i).p3 == pyramids(j).p2 ||
                    pyramids(i).p3 == pyramids(j).p3 || pyramids(i).p3 == pyramids(j).p4)
                {
                    count++;
                }
                if (pyramids(i).p4 == pyramids(j).p1 || pyramids(i).p4 == pyramids(j).p2 ||
                    pyramids(i).p4 == pyramids(j).p3 || pyramids(i).p4 == pyramids(j).p4)
                {
                    count++;
                }
                if (count == 3)
                {
                    int index = Kokkos::atomic_fetch_add(&(adjacentPyramids.value_at(i).adjacentCount_3), 1);
                    if (index < MAX_NEIGHBOR_COUNT)
                    {
                        adjacentPyramids.value_at(i).adjacentPyramids_3[index] = j;
                    }
                    else
                    {
                        Kokkos::printf("Neighbor count exceeds the limit\n");
                    }
                }
                else if (count == 2)
                {
                    int index = Kokkos::atomic_fetch_add(&(adjacentPyramids.value_at(i).adjacentCount_2), 1);
                    if (index < MAX_NEIGHBOR_COUNT_2)
                    {
                        adjacentPyramids.value_at(i).adjacentPyramids_2[index] = j;
                    }
                    else
                    {
                        Kokkos::printf("Neighbor count exceeds the limit\n");
                    }
                }
                else if (count == 1)
                {
                    int index = Kokkos::atomic_fetch_add(&(adjacentPyramids.value_at(i).adjacentCount_1), 1);
                    if (index < MAX_NEIGHBOR_COUNT_4)
                    {
                        adjacentPyramids.value_at(i).adjacentPyramids_1[index] = j;
                    }
                    else
                    {
                        Kokkos::printf("Neighbor count exceeds the limit\n");
                    }
                }
            });
    }
    TetMesh(const std::string &filename) : pyramids(Kokkos::View<Pyramid *, ExecSpace>("pyramids", 0))
    {
        load_from_file(filename);
        buildNeighbors();
    }
};
#endif