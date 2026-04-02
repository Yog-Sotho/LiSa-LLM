#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

using json = nlohmann::json;

// ---------------------------------------------------------------------
// Simple JSON logger (one line per log entry)
// ---------------------------------------------------------------------
inline void log_json(const json& j) {
    std::cout << j.dump() << std::endl;
}

// ---------------------------------------------------------------------
// Configuration structure (read from config.yaml)
// ---------------------------------------------------------------------
struct Config {
    std::string model_path;
    std::string vocab_path;          // BPE vocab JSON
    std::string listen_addr = "0.0.0.0";
    int listen_port = 8080;
    int n_threads = 4;
    size_t max_context = 2048;
    size_t memory_limit_bytes = 2ULL << 30; // 2 GB default
};

inline Config load_config(const std::string& file) {
    Config cfg;
    YAML::Node root = YAML::LoadFile(file);
    cfg.model_path = root["model_path"].as<std::string>();
    cfg.vocab_path = root["vocab_path"].as<std::string>();
    if (root["listen_addr"]) cfg.listen_addr = root["listen_addr"].as<std::string>();
    if (root["listen_port"]) cfg.listen_port = root["listen_port"].as<int>();
    if (root["n_threads"]) cfg.n_threads = root["n_threads"].as<int>();
    if (root["max_context"]) cfg.max_context = root["max_context"].as<size_t>();
    if (root["memory_limit_bytes"]) cfg.memory_limit_bytes = root["memory_limit_bytes"].as<size_t>();
    return cfg;
}

// ---------------------------------------------------------------------
// ISO‑8601 timestamp helper (used in logs)
// ---------------------------------------------------------------------
inline std::string iso8601_now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%FT%TZ", &tm);
    return std::string(buf);
}
