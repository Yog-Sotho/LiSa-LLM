#pragma once

#include <cstdint>
#include <string>
#include <string_view>

// Isolation for the inference worker, the only process that parses the (untrusted) model
// file. See docs/project/threat_model.md for what it does and does not protect against.
namespace lisa {

// Where the model file appears inside the worker's root.
inline constexpr std::string_view kSandboxModelPath = "/model.gguf";

struct SandboxPolicy {
    std::uint64_t memory_limit_bytes = 0;  // RLIMIT_DATA (and memory.max with a cgroup)
    std::string cgroup_parent;             // delegated cgroup v2 directory, or empty
};

enum class SeccompStage : std::uint8_t {
    load,   // model loading: may still open the model file
    serve,  // after loading: no file access at all
};

// True when the binary was built with LISA_SANDBOX=ON (Linux + libseccomp).
[[nodiscard]] bool sandbox_available() noexcept;

// Locks the calling process down. It must be single-threaded. Afterwards it runs as PID 1
// of new user, mount, network, IPC, UTS, cgroup and PID namespaces, with an empty
// read-only root containing only `model_fd` (read-only) at kSandboxModelPath, no
// capabilities, no_new_privs, Landlock and resource limits. A small reaper process stays
// outside the PID namespace and exits with the worker's status (128 + signal when it was
// killed). Throws std::runtime_error, with a hint when the host forbids namespaces.
void enter_worker_sandbox(const SandboxPolicy& policy, int model_fd);

// Installs a seccomp-bpf allow-list for all threads; any other system call kills the
// process (logged instead when `log_only`). Stages stack, so serve narrows load.
void install_seccomp(SeccompStage stage, bool log_only);

// Landlock ABI applied by enter_worker_sandbox (0 when the kernel lacks Landlock).
[[nodiscard]] int landlock_abi_in_effect() noexcept;

}  // namespace lisa
