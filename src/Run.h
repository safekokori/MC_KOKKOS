#ifndef RUN_H
#define RUN_H
#include "Transpose_core.h"

class Run
{
   public:
    Run(TetMesh mesh) : m_mesh(mesh) {}
    Kokkos::View<resultType*, Kokkos::HostSpace> run(unsigned int num_photons)
    {
        Kokkos::View<resultType*, ExecSpace> results("results", num_photons);
        auto rand_pool = RandPoolType(time(NULL));
        auto strategy = DefaultEmitCollectStrategy(m_mesh);

        Kokkos::parallel_for(
            "run", num_photons,
            KOKKOS_LAMBDA(const unsigned int i)
            {
                transpose_core core(m_mesh, strategy, rand_pool);
                core.run();
                results(i) = core.result;
            }
        );
        Kokkos::View<resultType*, Kokkos::HostSpace> host_results("results", num_photons);
        Kokkos::deep_copy(host_results, results);
        return host_results;
    };

   private:
    TetMesh m_mesh;
};
#endif