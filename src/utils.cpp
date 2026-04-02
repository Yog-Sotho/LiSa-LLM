#include "utils.hpp"

Config load_config(const std::string& filename) {
    YAML::Node root = YAML::LoadFile(filename);
    Config cfg;
    cfg.model_path = root["model_path"].as<std::string>();
    cfg.vocab_path = root["vocab_path"].as<std::string>();
    cfg.merges_path = root["merges_path"].as<std::string>();
    if (root["listen_addr"]) cfg.listen_addr = root["listen_addr"].as<std::string>();
    if (root["listen_port"]) cfg.listen_port = root["listen_port"].as<int>();
    if (root["enable_tls"]) cfg.enable_tls = root["enable_tls"].as<bool>();
    if (root["cert_file"]) cfg.cert_file = root["cert_file"].as<std::string>();
    if (root["key_file"]) cfg.key_file = root["key_file"].as<std::string>();
    if (root["api_key"]) cfg.api_key = root["api_key"].as<std::string>();
    if (root["n_threads"]) cfg.n_threads = root["n_threads"].as<int>();
    if (root["max_context"]) cfg.max_context = root["max_context"].as<size_t>();
    if (root["temperature_default"]) cfg.temperature_default = root["temperature_default"].as<float>();
    if (root["top_p_default"]) cfg.top_p_default = root["top_p_default"].as<float>();
    if (root["max_new_tokens_default"]) cfg.max_new_tokens_default = root["max_new_tokens_default"].as<int>();
    if (root["memory_limit_bytes"]) cfg.memory_limit_bytes = root["memory_limit_bytes"].as<size_t>();
    if (root["sandbox_enabled"]) cfg.sandbox_enabled = root["sandbox_enabled"].as<bool>();
    if (root["use_gpu"]) cfg.use_gpu = root["use_gpu"].as<bool>();
    if (root["gpu_layers"]) cfg.gpu_layers = root["gpu_layers"].as<int>();
    return cfg;
}
