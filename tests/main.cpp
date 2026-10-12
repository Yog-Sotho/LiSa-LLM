// Test runner entry point. The test binary doubles as the sandboxed worker (SandboxedEngine
// re-executes /proc/self/exe), with escape probes available to the sandbox tests.

#include "worker.hpp"

#include <catch2/catch_session.hpp>
#include <string_view>

lisa::json lisa_test_probe(const lisa::json& request);  // sandbox_probes.cpp

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == lisa::kWorkerFlag) {
        return lisa::worker_main(&lisa_test_probe);
    }
    return Catch::Session().run(argc, argv);
}
