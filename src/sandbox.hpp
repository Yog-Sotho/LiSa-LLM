#pragma once

#include "utils.hpp"

namespace lisa {

// True when the binary was built with LISA_SANDBOX=ON.
[[nodiscard]] bool sandbox_available() noexcept;

// Sets up the sandbox in the calling (child) process. Returns 0 on success; exits the
// process on unrecoverable failures.
//
// WARNING: experimental. The current implementation is known not to isolate the server
// (see docs/project/codebase_audit.md, roadmap P2) and is disabled by default.
int sandbox_init(const Config& cfg);

}  // namespace lisa
