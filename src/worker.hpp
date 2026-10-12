#pragma once

#include "engine.hpp"
#include "ipc.hpp"
#include "utils.hpp"

#include <string_view>

namespace lisa {

// Command-line flag that turns the lisa binary into the sandboxed inference worker. The
// broker starts it with the IPC socket on fd 3 and the opened model file on fd 4.
inline constexpr std::string_view kWorkerFlag = "--sandbox-worker";
inline constexpr int kWorkerChannelFd = 3;
inline constexpr int kWorkerModelFd = 4;

// Test hook: answers {"op":"probe",...} requests from inside the sandbox. Production
// workers have none, so the op is rejected.
using ProbeHandler = json (*)(const json& request);

// Entry point of `lisa --sandbox-worker`: reads the settings, locks the process down,
// loads the model and serves requests until the broker hangs up.
int worker_main(ProbeHandler probe = nullptr);

// Serves broker requests on `channel` until it is closed. Returns 0 on a clean hang-up.
int serve_requests(ipc::Channel& channel, Engine& engine, ProbeHandler probe);

}  // namespace lisa
