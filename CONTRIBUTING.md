# Contributing to LiSa-LLM

Thanks for helping. Bug reports, fixes and features are welcome; for security issues use
[SECURITY.md](SECURITY.md) instead of a public issue.

## Set up

Linux with GCC 13+ or Clang 18+, CMake 3.28+, Ninja, OpenSSL 3, libseccomp and yaml-cpp
(package names per distribution: [Installation](docs/installation.md#build-from-source)). llama.cpp, cpp-httplib, nlohmann/json and Catch2 are
fetched by CMake, pinned by tag and verified commit.

```bash
cmake --preset dev && cmake --build --preset dev
curl -L -o /tmp/stories260K.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
LISA_TEST_MODEL=/tmp/stories260K.gguf LISA_REQUIRE_SANDBOX=1 ctest --preset dev
```

The sandbox tests need unprivileged user namespaces (on Ubuntu 23.10+:
`sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`). Without
`LISA_REQUIRE_SANDBOX=1` they skip where namespaces are blocked; CI sets it so they never
skip silently.

## Presets

| Preset | Use |
|---|---|
| `dev` | Debug, warnings as errors |
| `asan-ubsan`, `tsan` | Sanitizers (TSan also instruments the engine; OpenMP off) |
| `coverage` | Coverage build; `gcovr --root . --object-directory build/coverage --filter src/` |
| `release` | Optimised, portable CPU flags |
| `portable` | Release binary for distribution (static deps except glibc) |
| `fuzz` | libFuzzer targets (clang): `CC=clang CXX=clang++ cmake --preset fuzz` |
| `cuda` | CUDA offload |

## Before you open a pull request

CI runs all of this; running it locally saves a round trip.

```bash
clang-format -i src/*.cpp src/*.hpp tests/*.cpp tests/*.hpp fuzz/*.cpp
CC=clang CXX=clang++ cmake --preset dev -B build/tidy -DLISA_NATIVE=OFF
clang-tidy -p build/tidy --quiet src/*.cpp tests/*.cpp
cppcheck --enable=warning,performance,portability,style --std=c++20 --inline-suppr \
  --suppress=missingIncludeSystem --suppress=unusedFunction -I src -q src tests fuzz
for p in dev asan-ubsan tsan; do cmake --build --preset $p && ctest --preset $p; done
actionlint && shellcheck packaging/*.sh     # when touching workflows or scripts
```

- **Tests**: new behaviour comes with tests. HTTP behaviour that depends on timing uses the
  scripted engine in `tests/fake_engine.hpp`; anything concurrent must pass under TSan.
- **Coverage**: CI fails below 85 % line coverage of `src/` (the worker-only `sandbox.cpp`
  and `worker.cpp` are excluded because the sandbox can't write coverage data; the
  sandbox tests check them from outside).
- **Sandbox changes**: a new system call in the worker needs a reason, the narrowest
  seccomp rule that works, a passing escape-probe suite, and a note in
  [the threat model](docs/project/threat_model.md).
- **Style**: follow the surrounding code: C++20, `.clang-format`, `.clang-tidy`; comments
  explain why, not what. Log with `log_info/warn/error` (structured JSON).
- **Commits**: small and focused, with a message that explains the change and its reason.
  Update `CHANGELOG.md` under `## [Unreleased]` for user-visible changes.

## Fuzzing and benchmarks

```bash
CC=clang CXX=clang++ cmake --preset fuzz && cmake --build --preset fuzz
build/fuzz/fuzz/fuzz_request -dict=fuzz/request.dict fuzz/corpus/request
build/fuzz/fuzz/fuzz_worker_reply fuzz/corpus/worker_reply
python3 bench/bench.py --url http://127.0.0.1:8080 --key "$KEY" --concurrency 1,2,4,8
```

## Dependencies

Third-party code is pinned in `cmake/Dependencies.cmake` (tag + commit, or URL + SHA-256),
container base images by digest in `Dockerfile` and the workflows, GitHub Actions by commit
SHA. Update them deliberately, one at a time, with the changelog of the dependency in mind.

## Releasing (maintainers)

1. Set the version in `project(LiSa-LLM VERSION X.Y.Z)` (CMakeLists.txt) and rename the
   changelog's `## [Unreleased]` section to `## [X.Y.Z]`.
2. Optionally run the *Release* workflow by hand: it builds and tests everything without
   publishing.
3. Tag and push: `git tag -s vX.Y.Z -m "LiSa-LLM X.Y.Z" && git push origin vX.Y.Z`.
   The workflow checks the tag against the version and changelog, then publishes the
   binaries, images, SBOMs, signatures and provenance and creates the GitHub release.
