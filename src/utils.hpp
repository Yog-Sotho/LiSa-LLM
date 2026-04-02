#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

using json = nlohmann::json;

inline void log_info(const std::string& msg, const json& extra = json::object()) {
    json j = {
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count()},
        {"level", "info"},
        {"message", msg}
    };
    j.merge_patch(extra);
    std::cout << j.dump() << std::endl;
}

inline void log_error(const std::string& msg, const json& extra = json::object()) {
    json j = {
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count()},
        {"level", "error"},
        {"message", msg}
    };
    j.merge_patch(extra);
    std::cerr << j.dump() << std::endl;
}

inline std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%FT%TZ", &tm);
    return buf;
}

struct Config {
    std::string model_path;
    std::string vocab_path;
    std::string merges_path;
    std::string listen_addr = "127.0.0.1";
    int listen_port = 8080;
    bool enable_tls = false;
    std::string cert_file;
    std::string key_file;
    std::string api_key;
    int n_threads = 4;
    size_t max_context = 2048;
    float temperature_default = 0.8f;
    float top_p_default = 0.95f;
    int max_new_tokens_default = 128;
    size_t memory_limit_bytes = 2ULL << 30;
    bool sandbox_enabled = true;
    bool use_gpu = true;
    int gpu_layers = 100;  // offload all layers to GPU
};

Config load_config(const std::string& filename);
