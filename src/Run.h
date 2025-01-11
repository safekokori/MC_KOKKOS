#ifndef RUN_H
#define RUN_H
#include "Kokkos_Assert.hpp"
#include "Transpose_core.h"

class Run
{
   public:
    Run(const char* mesh_path) : m_mesh_path(mesh_path), m_mesh(mesh_path) {}
    Kokkos::View<resultType*, Kokkos::HostSpace> run(unsigned int num_photons)
    {
        check_Mesh();
        Kokkos::View<resultType*, ExecSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>> results("results",
                                                                                                 num_photons);
        auto rand_pool = RandPoolType(time(NULL));
        Kokkos::UnorderedMap<Index, CollectType, ExecSpace> collect_map(m_mesh.pyramids.extent(0));
        auto strategy = DefaultCollectStrategy(collect_map);
        bool log      = false;
        if (num_photons == 1)
        {
            log = true;
        }
        if (log)
        {
            Kokkos::printf("log is on\n");
        }
        Kokkos::parallel_for(
            "run", Kokkos::RangePolicy<ExecSpace>(0, num_photons),
            KOKKOS_CLASS_LAMBDA(const unsigned int i)
            {
                transpose_core core(m_mesh, strategy, rand_pool);
                core.run(log);
                results(i) = core.result;
            });
        Kokkos::View<resultType*, Kokkos::HostSpace> host_results("results", num_photons);
        Kokkos::deep_copy(host_results, results);
        return host_results;
    };
    void check_Mesh()
    {
#if defined(NDEBUG) and not defined(KOKKOS_ENABLE_DEBUG)
        Kokkos::abort("KOKKOS_ASSERT NOT DEFINED!!!");
#endif

        KOKKOS_ASSERT(m_mesh.pyramids.extent(0) > 0);
        auto policy = Kokkos::RangePolicy<>(0, m_mesh.pyramids.extent(0));
        Kokkos::parallel_for(
            "check_Mesh", policy, KOKKOS_CLASS_LAMBDA(const int i) {
                KOKKOS_ASSERT(!IsNan(m_mesh.pyramids(i).value.mua) && !IsNan(m_mesh.pyramids(i).value.mus) &&
                              !IsNan(m_mesh.pyramids(i).value.g) && !IsNan(m_mesh.pyramids(i).value.n) &&
                              "mesh属性中存在nan值");
            });
    }

   private:
    const char* m_mesh_path;
    TetMesh m_mesh;
};
#endif