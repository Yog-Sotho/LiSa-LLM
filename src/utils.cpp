#include "utils.hpp"

#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace lisa {

namespace {

constexpr int kMaxPort = 65535;

template <typename T>
void read_optional(const YAML::Node& root, const char* key, T& out) {
    if (const YAML::Node node = root[key]) {
        out = node.as<T>();
    }
}

void validate(const Config& cfg) {
    if (cfg.model_path.empty()) {
        throw std::runtime_error("model_path is required");
    }
    if (cfg.listen_port < 1 || cfg.listen_port > kMaxPort) {
        throw std::runtime_error("listen_port must be between 1 and 65535");
    }
    if (cfg.n_threads < 1) {
        throw std::runtime_error("n_threads must be at least 1");
    }
    if (cfg.max_context < 1) {
        throw std::runtime_error("max_context must be at least 1");
    }
    if (cfg.max_new_tokens_default < 1) {
        throw std::runtime_error("max_new_tokens_default must be at least 1");
    }
    if (cfg.enable_tls && (cfg.cert_file.empty() || cfg.key_file.empty())) {
        throw std::runtime_error("enable_tls requires cert_file and key_file");
    }
}

}  // namespace

Config load_config(const std::string& filename) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(filename);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("cannot read config '" + filename + "': " + e.what());
    }
    if (!root.IsMap()) {
        throw std::runtime_error("config '" + filename + "' must be a YAML mapping");
    }

    Config cfg;
    try {
        read_optional(root, "model_path", cfg.model_path);
        read_optional(root, "listen_addr", cfg.listen_addr);
        read_optional(root, "listen_port", cfg.listen_port);
        read_optional(root, "enable_tls", cfg.enable_tls);
        read_optional(root, "cert_file", cfg.cert_file);
        read_optional(root, "key_file", cfg.key_file);
        read_optional(root, "api_key", cfg.api_key);
        read_optional(root, "n_threads", cfg.n_threads);
        read_optional(root, "max_context", cfg.max_context);
        read_optional(root, "temperature_default", cfg.temperature_default);
        read_optional(root, "top_p_default", cfg.top_p_default);
        read_optional(root, "max_new_tokens_default", cfg.max_new_tokens_default);
        read_optional(root, "memory_limit_bytes", cfg.memory_limit_bytes);
        read_optional(root, "sandbox_enabled", cfg.sandbox_enabled);
        read_optional(root, "use_gpu", cfg.use_gpu);
        read_optional(root, "gpu_layers", cfg.gpu_layers);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("invalid value in config '" + filename + "': " + e.what());
    }

    for (const char* key : {"vocab_path", "merges_path"}) {
        if (root[key]) {
            log_warn("config key is obsolete and ignored: the tokenizer is read from the GGUF file",
                     {{"key", key}});
        }
    }

    validate(cfg);
    return cfg;
}

bool is_loopback_address(std::string_view addr) noexcept {
    return addr == "localhost" || addr == "::1" || addr.starts_with("127.");
}

}  // namespace lisa
