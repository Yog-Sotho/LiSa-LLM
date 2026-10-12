#include "utils.hpp"

#include <sched.h>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>

namespace lisa {

namespace {

constexpr int kMaxPort = 65535;
constexpr std::size_t kSha256HexDigits = 64;
constexpr std::string_view kSha256Prefix = "sha256:";
constexpr std::string_view kPlaceholderApiKey = "changeme";
constexpr std::size_t kMinExposedKeyLength = 16;
constexpr int kMaxQueuedRequests = 4096;
constexpr int kMaxTimeoutSeconds = 3600;
constexpr mode_t kGroupOtherBits = 077;

using Setter = std::function<void(Config&, const YAML::Node&)>;

template <typename T>
Setter field(T Config::*member) {
    return [member](Config& cfg, const YAML::Node& node) { cfg.*member = node.as<T>(); };
}

// Every accepted key. Anything else in a config file is an error, so typos don't go
// unnoticed.
const std::map<std::string, Setter, std::less<>>& fields() {
    static const std::map<std::string, Setter, std::less<>> table = {
        {"model_path", field(&Config::model_path)},
        {"model_alias", field(&Config::model_alias)},
        {"chat_template", field(&Config::chat_template)},
        {"listen_addr", field(&Config::listen_addr)},
        {"listen_port", field(&Config::listen_port)},
        {"enable_tls", field(&Config::enable_tls)},
        {"cert_file", field(&Config::cert_file)},
        {"key_file", field(&Config::key_file)},
        {"client_ca_file", field(&Config::client_ca_file)},
        {"tls_min_version", field(&Config::tls_min_version)},
        {"api_key", field(&Config::api_key)},
        {"api_keys", field(&Config::api_keys)},
        {"api_keys_file", field(&Config::api_keys_file)},
        {"max_queued_requests", field(&Config::max_queued_requests)},
        {"rate_limit_per_minute", field(&Config::rate_limit_per_minute)},
        {"rate_limit_burst", field(&Config::rate_limit_burst)},
        {"max_prompt_tokens", field(&Config::max_prompt_tokens)},
        {"read_timeout_seconds", field(&Config::read_timeout_seconds)},
        {"write_timeout_seconds", field(&Config::write_timeout_seconds)},
        {"keep_alive_timeout_seconds", field(&Config::keep_alive_timeout_seconds)},
        {"shutdown_timeout_seconds", field(&Config::shutdown_timeout_seconds)},
        {"n_threads", field(&Config::n_threads)},
        {"max_context", field(&Config::max_context)},
        {"temperature_default", field(&Config::temperature_default)},
        {"top_p_default", field(&Config::top_p_default)},
        {"max_new_tokens_default", field(&Config::max_new_tokens_default)},
        {"model_sha256", field(&Config::model_sha256)},
        {"sandbox_enabled", field(&Config::sandbox_enabled)},
        {"memory_limit_bytes", field(&Config::memory_limit_bytes)},
        {"sandbox_cgroup", field(&Config::sandbox_cgroup)},
        {"sandbox_seccomp_log", field(&Config::sandbox_seccomp_log)},
        {"use_gpu", field(&Config::use_gpu)},
        {"gpu_layers", field(&Config::gpu_layers)},
    };
    return table;
}

bool is_obsolete_key(std::string_view key) {
    return key == "vocab_path" || key == "merges_path";
}

std::string env_name(std::string_view key) {
    std::string name = "LISA_";
    std::transform(key.begin(), key.end(), std::back_inserter(name),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return name;
}

// An environment value as a YAML node: the raw string (never parsed as YAML, so "#" or
// ":" in a key stay literal), or a sequence split on commas for list keys.
YAML::Node env_node(std::string_view key, const std::string& value) {
    if (key != "api_keys") {
        return YAML::Node(value);
    }
    YAML::Node list(YAML::NodeType::Sequence);
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = std::min(value.find(',', start), value.size());
        if (comma > start) {
            list.push_back(value.substr(start, comma - start));
        }
        start = comma + 1;
    }
    return list;
}

void apply(Config& cfg, const std::string& key, const YAML::Node& node, const std::string& origin) {
    try {
        fields().at(key)(cfg, node);
    } catch (const YAML::Exception&) {
        throw std::runtime_error("invalid value for '" + key + "' in " + origin);
    }
}

std::string trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(first, last - first + 1));
}

// One key per line; blank lines and lines starting with '#' are ignored.
void read_api_keys_file(Config& cfg) {
    std::ifstream in(cfg.api_keys_file);
    if (!in) {
        throw std::runtime_error("cannot read api_keys_file '" + cfg.api_keys_file + "'");
    }
    bool plaintext = false;
    for (std::string line; std::getline(in, line);) {
        std::string entry = trim(line);
        if (entry.empty() || entry.starts_with('#')) {
            continue;
        }
        plaintext = plaintext || !entry.starts_with(kSha256Prefix);
        cfg.api_keys.push_back(std::move(entry));
    }
    struct stat st {};
    if (plaintext && ::stat(cfg.api_keys_file.c_str(), &st) == 0 &&
        (st.st_mode & kGroupOtherBits) != 0) {
        log_warn(
            "api_keys_file holds plaintext keys and is readable by other users; "
            "chmod 600 it or store sha256: digests (lisa --hash-key)",
            {{"file", cfg.api_keys_file}});
    }
}

bool is_hex_digest(std::string_view text) {
    return text.size() == kSha256HexDigits &&
           text.find_first_not_of("0123456789abcdefABCDEF") == std::string_view::npos;
}

void require_range(int value, int min, int max, const char* name) {
    if (value < min || value > max) {
        throw std::runtime_error(std::string(name) + " must be between " + std::to_string(min) +
                                 " and " + std::to_string(max));
    }
}

void validate(const Config& cfg) {
    if (cfg.model_path.empty()) {
        throw std::runtime_error("model_path is required");
    }
    require_range(cfg.listen_port, 1, kMaxPort, "listen_port");
    if (cfg.n_threads < 0) {
        throw std::runtime_error("n_threads must not be negative (0 = one per CPU)");
    }
    if (cfg.max_context < 1) {
        throw std::runtime_error("max_context must be at least 1");
    }
    if (cfg.max_new_tokens_default < 1) {
        throw std::runtime_error("max_new_tokens_default must be at least 1");
    }
    if (!cfg.model_sha256.empty() && !is_hex_digest(cfg.model_sha256)) {
        throw std::runtime_error("model_sha256 must be 64 hexadecimal digits");
    }
    if (cfg.enable_tls && (cfg.cert_file.empty() || cfg.key_file.empty())) {
        throw std::runtime_error("enable_tls requires cert_file and key_file");
    }
    if (!cfg.client_ca_file.empty() && !cfg.enable_tls) {
        throw std::runtime_error("client_ca_file requires enable_tls");
    }
    if (cfg.tls_min_version != "1.2" && cfg.tls_min_version != "1.3") {
        throw std::runtime_error(R"(tls_min_version must be "1.2" or "1.3")");
    }
    for (const std::string& key : cfg.api_keys) {
        if (key.empty()) {
            throw std::runtime_error("api_keys must not contain empty keys");
        }
        if (key.starts_with(kSha256Prefix) && !is_hex_digest(key.substr(kSha256Prefix.size()))) {
            throw std::runtime_error(
                "an api_keys entry starting with sha256: must be followed "
                "by 64 hexadecimal digits");
        }
    }
    require_range(cfg.max_queued_requests, 0, kMaxQueuedRequests, "max_queued_requests");
    require_range(cfg.rate_limit_per_minute, 0, 1'000'000, "rate_limit_per_minute");
    require_range(cfg.rate_limit_burst, 1, 1'000'000, "rate_limit_burst");
    if (cfg.max_prompt_tokens < 0) {
        throw std::runtime_error("max_prompt_tokens must not be negative");
    }
    require_range(cfg.read_timeout_seconds, 1, kMaxTimeoutSeconds, "read_timeout_seconds");
    require_range(cfg.write_timeout_seconds, 1, kMaxTimeoutSeconds, "write_timeout_seconds");
    require_range(cfg.keep_alive_timeout_seconds, 1, kMaxTimeoutSeconds,
                  "keep_alive_timeout_seconds");
    require_range(cfg.shutdown_timeout_seconds, 0, kMaxTimeoutSeconds, "shutdown_timeout_seconds");
}

}  // namespace

EnvLookup process_environment() {
    return [](const std::string& name) -> std::optional<std::string> {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): read once at startup, before any thread
        const char* value = std::getenv(name.c_str());
        if (value == nullptr) {
            return std::nullopt;
        }
        return std::string(value);
    };
}

Config load_config(const std::string& filename, const EnvLookup& env,
                   const ConfigOverrides& overrides) {
    Config cfg;
    if (!filename.empty()) {
        YAML::Node root;
        try {
            root = YAML::LoadFile(filename);
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("cannot read config '" + filename + "': " + e.what());
        }
        if (root.IsNull()) {
            root = YAML::Node(YAML::NodeType::Map);  // an empty file
        }
        if (!root.IsMap()) {
            throw std::runtime_error("config '" + filename + "' must be a YAML mapping");
        }
        const std::string origin = "config '" + filename + "'";
        for (const auto& entry : root) {
            const auto key = entry.first.as<std::string>();
            if (is_obsolete_key(key)) {
                log_warn(
                    "config key is obsolete and ignored: the tokenizer is read from the "
                    "GGUF file",
                    {{"key", key}});
            } else if (fields().contains(key)) {
                apply(cfg, key, entry.second, origin);
            } else {
                std::string message = "unknown key '" + key;
                message += "' in ";
                message += origin;
                throw std::runtime_error(message);
            }
        }
    }

    for (const auto& [key, setter] : fields()) {
        const std::string name = env_name(key);
        if (const std::optional<std::string> value = env(name)) {
            apply(cfg, key, env_node(key, *value), "environment variable " + name);
        }
    }

    if (overrides.model_path) {
        cfg.model_path = *overrides.model_path;
    }
    if (overrides.listen_addr) {
        cfg.listen_addr = *overrides.listen_addr;
    }
    if (overrides.listen_port) {
        cfg.listen_port = *overrides.listen_port;
    }
    if (!cfg.api_keys_file.empty()) {
        read_api_keys_file(cfg);
    }

    validate(cfg);
    const int cpus = available_cpus();
    if (cfg.n_threads == 0) {
        cfg.n_threads = cpus;
    } else if (cfg.n_threads > cpus) {
        // Extra threads spin against each other: generation gets several times slower.
        log_warn("n_threads is larger than the number of available CPUs; set it to 0 (auto)",
                 {{"n_threads", cfg.n_threads}, {"cpus", cpus}});
    }
    return cfg;
}

std::string exposure_problem(const Config& cfg) {
    if (is_loopback_address(cfg.listen_addr)) {
        return {};
    }
    std::vector<std::string> plaintext = cfg.api_keys;
    if (!cfg.api_key.empty()) {
        plaintext.push_back(cfg.api_key);
    }
    for (const std::string& key : plaintext) {
        if (key.starts_with(kSha256Prefix)) {
            continue;  // a digest says nothing about the key's strength
        }
        if (key == kPlaceholderApiKey) {
            return "the placeholder API key 'changeme' is configured";
        }
        if (key.size() < kMinExposedKeyLength) {
            std::string problem = "an API key is shorter than ";
            problem += std::to_string(kMinExposedKeyLength);
            problem += " characters";
            return problem;
        }
    }
    return {};
}

bool constant_time_equals(std::string_view given, std::string_view expected) noexcept {
    std::size_t diff = given.size() ^ expected.size();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto g = i < given.size() ? static_cast<unsigned char>(given[i]) : 0U;
        diff |= g ^ static_cast<unsigned char>(expected[i]);
    }
    return diff == 0;
}

int available_cpus() noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof set, &set) == 0) {
        return std::max(1, CPU_COUNT(&set));
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

bool is_loopback_address(std::string_view addr) noexcept {
    return addr == "localhost" || addr == "::1" || addr.starts_with("127.");
}

}  // namespace lisa
