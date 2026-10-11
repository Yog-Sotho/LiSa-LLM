#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

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
    std::string listen_addr = "127.0.0.1";
    int listen_port = 8080;
    bool enable_tls = false;
    std::string cert_file;
    std::string key_file;
    std::string api_key;
    int n_threads = 4;
    std::uint32_t max_context = 2048;
    float temperature_default = 0.8f;
    float top_p_default = 0.95f;
    int max_new_tokens_default = 128;
    std::uint64_t memory_limit_bytes = 2ULL << 30;
    // The sandbox is being redesigned (roadmap P2) and does not isolate the server yet.
    bool sandbox_enabled = false;
    bool use_gpu = true;
    int gpu_layers = -1;  // -1 = offload every layer when built with CUDA
};

// Loads and validates a YAML config file. Throws std::runtime_error with a readable
// message on I/O, syntax or validation errors.
[[nodiscard]] Config load_config(const std::string& filename);

// True when `addr` only accepts connections from the local machine.
[[nodiscard]] bool is_loopback_address(std::string_view addr) noexcept;

}  // namespace lisa
