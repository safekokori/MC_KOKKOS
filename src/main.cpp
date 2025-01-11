#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <fstream>
#include "Utils.h"
#include "Geometry.h"
#include "Run.h"

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);
    Kokkos::printf("DefaultExecutionSpace: %s HostSpace: %s\n", ExecSpace::name(), Kokkos::HostSpace::name());
    TetMesh mesh("data/MultiLayers.vol");

    Run run("data/MultiLayers.vol");
    auto res = run.run(1);
    for (int i = 0; i < 1; i++)
    {
        printf("%d\n", (int)res(i).type);
    }
    return 0;
}