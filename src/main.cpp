#include "Run.h"

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);
    Kokkos::printf("%s\n", Kokkos::DefaultExecutionSpace::name());
    TetMesh mesh("data/bunny.vol");
    Run run(mesh);
    run.run(1000);
    return 0;
}