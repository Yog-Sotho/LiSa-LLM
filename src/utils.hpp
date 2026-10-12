#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lisa {

using json = nlohmann::json;

namespace detail {
inline void log_line(std::ostream& out, std::string_view level, const std::string& msg,
                     const json& extra) {
    json j = {{"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count()},
              {"level", level},
              {"message", msg}};
    j.merge_patch(extra);
    // `replace` keeps logging from throwing on invalid UTF-8 coming from model output.
    out << j.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
    out.flush();  // one complete line per event, even when stdout is a pipe
}
}  // namespace detail

inline void log_info(const std::string& msg, const json& extra = json::object()) {
    detail::log_line(std::cout, "info", msg, extra);
}

inline void log_warn(const std::string& msg, const json& extra = json::object()) {
    detail::log_line(std::cerr, "warn", msg, extra);
}

inline void log_error(const std::string& msg, const json& extra = json::object()) {
    detail::log_line(std::cerr, "error", msg, extra);
}

struct Config {
    std::string model_path;
    std::string model_alias;    // model id reported by the API; default: model file name
    std::string chat_template;  // built-in template name; empty = from the model (else ChatML)
    std::string listen_addr = "127.0.0.1";
    int listen_port = 8080;

    bool enable_tls = false;
    std::string cert_file;
    std::string key_file;
    std::string client_ca_file;           // non-empty: require client certificates (mTLS)
    std::string tls_min_version = "1.2";  // "1.2" or "1.3"

    // Accepted API keys: api_key, api_keys, the lines of api_keys_file and LISA_API_KEYS.
    // An entry is either the key itself or "sha256:<hex digest of the key>".
    std::string api_key;
    std::vector<std::string> api_keys;
    std::string api_keys_file;

    // Request admission and timeouts.
    int max_queued_requests = 16;    // requests waiting for the model; more get 429
    int rate_limit_per_minute = 0;   // per API key (or client address); 0 = unlimited
    int rate_limit_burst = 10;       // requests a key may send at once
    int max_prompt_tokens = 0;       // 0 = only the context window limits prompts
    int read_timeout_seconds = 30;   // receiving a request
    int write_timeout_seconds = 60;  // sending each part of a response
    int keep_alive_timeout_seconds = 5;
    int shutdown_timeout_seconds = 30;  // how long SIGTERM waits for running requests

    int n_threads = 0;  // 0 = one per CPU the process may run on
    std::uint32_t max_context = 2048;
    float temperature_default = 0.8f;
    float top_p_default = 0.95f;
    int max_new_tokens_default = 128;
    std::string model_sha256;  // optional hex SHA-256 the model file must match
    // Run the model in an isolated worker process (namespaces, Landlock, seccomp).
    bool sandbox_enabled = true;
    std::uint64_t memory_limit_bytes = 4ULL << 30;  // worker heap limit (sandbox only)
    std::string sandbox_cgroup;                     // optional delegated cgroup v2 directory
    bool sandbox_seccomp_log = false;               // log seccomp violations instead of killing
    bool use_gpu = true;
    int gpu_layers = -1;  // -1 = offload every layer when built with CUDA
};

// Looks up an environment variable; std::nullopt when unset.
using EnvLookup = std::function<std::optional<std::string>(const std::string&)>;

// The process environment.
[[nodiscard]] EnvLookup process_environment();

// Values given on the command line; they win over the file and the environment.
struct ConfigOverrides {
    std::optional<std::string> model_path;
    std::optional<std::string> listen_addr;
    std::optional<int> listen_port;
};

// Loads a YAML config file (an empty name means "no file, defaults only"), applies
// LISA_<KEY> environment variables (e.g. LISA_LISTEN_PORT, LISA_API_KEYS=a,b) and then
// `overrides`, reads api_keys_file and validates the result. Unknown keys are errors.
// Throws std::runtime_error with a readable message.
[[nodiscard]] Config load_config(const std::string& filename,
                                 const EnvLookup& env = process_environment(),
                                 const ConfigOverrides& overrides = {});

// Why this configuration must not be exposed beyond the local machine (a placeholder or
// short API key on a non-loopback address), or an empty string when it may.
[[nodiscard]] std::string exposure_problem(const Config& cfg);

// Compares a client-supplied secret with the expected one in time that depends only on
// the expected secret's length.
[[nodiscard]] bool constant_time_equals(std::string_view given, std::string_view expected) noexcept;

// CPUs this process may run on (its affinity mask), at least 1.
[[nodiscard]] int available_cpus() noexcept;

// True when `addr` only accepts connections from the local machine.
[[nodiscard]] bool is_loopback_address(std::string_view addr) noexcept;

}  // namespace lisa
