#pragma once

#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// -----------------------------------------------------------------------------
// Logging: JSON lines to stdout
// -----------------------------------------------------------------------------
inline void log_info(const std::string& msg, const json& extra = json::object()) {
    json j = {
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count()},
        {"level", "info"},
        {"message", msg}
    };
    j.merge_patch(extra);
    std::cout << j.dump() << std::endl;
}

inline void log_error(const std::string& msg, const json& extra = json::object()) {
    json j = {
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count()},
        {"level", "error"},
        {"message", msg}
    };
    j.merge_patch(extra);
    std::cerr << j.dump() << std::endl;
}

// -----------------------------------------------------------------------------
// ISO 8601 timestamp (for HTTP Date headers)
// -----------------------------------------------------------------------------
inline std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%FT%TZ", &tm);
    return buf;
}

// -----------------------------------------------------------------------------
// Configuration (loaded from config.yaml)
// -----------------------------------------------------------------------------
struct Config {
    // Model
    std::string model_path;
    std::string vocab_path;
    std::string merges_path;        // BPE merges file (optional, for tokenizer)
    
    // Server
    std::string listen_addr = "127.0.0.1";
    int listen_port = 8080;
    bool enable_tls = false;
    std::string cert_file;
    std::string key_file;
    std::string api_key;            // if non‑empty, require X-API-Key header
    
    // Inference
    int n_threads = 4;
    size_t max_context = 2048;
    float temperature_default = 0.8f;
    float top_p_default = 0.95f;
    int max_new_tokens_default = 128;
    
    // Sandbox
    size_t memory_limit_bytes = 2ULL << 30;   // 2 GiB
    bool sandbox_enabled = true;
};

Config load_config(const std::string& filename);
