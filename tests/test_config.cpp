#include "utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Writes `content` to a unique temporary file that is removed on destruction.
class TempConfig {
public:
    explicit TempConfig(const std::string& content)
        : path_(std::filesystem::temp_directory_path() /
                ("lisa_test_" + std::to_string(counter_++) + ".yaml")) {
        std::ofstream(path_) << content;
    }
    ~TempConfig() { std::filesystem::remove(path_); }
    TempConfig(const TempConfig&) = delete;
    TempConfig& operator=(const TempConfig&) = delete;
    TempConfig(TempConfig&&) = delete;
    TempConfig& operator=(TempConfig&&) = delete;

    [[nodiscard]] std::string path() const { return path_.string(); }

private:
    static inline int counter_ = 0;
    std::filesystem::path path_;
};

// A fixed environment, so tests don't depend on the one ctest runs in.
lisa::EnvLookup fake_env(std::map<std::string, std::string> vars = {}) {
    return [vars = std::move(vars)](const std::string& name) -> std::optional<std::string> {
        const auto it = vars.find(name);
        if (it == vars.end()) {
            return std::nullopt;
        }
        return it->second;
    };
}

}  // namespace

TEST_CASE("load_config applies defaults for optional keys", "[config]") {
    const TempConfig file("model_path: ./m.gguf\n");
    const lisa::Config cfg = lisa::load_config(file.path());

    CHECK(cfg.model_path == "./m.gguf");
    CHECK(cfg.listen_addr == "127.0.0.1");
    CHECK(cfg.listen_port == 8080);
    CHECK_FALSE(cfg.enable_tls);
    CHECK(cfg.sandbox_enabled);  // secure by default
    CHECK(cfg.gpu_layers == -1);
}

TEST_CASE("load_config reads every supported key", "[config]") {
    const TempConfig file(R"(
model_path: /models/x.gguf
model_alias: my-model
chat_template: llama3
listen_addr: 0.0.0.0
listen_port: 9000
api_key: secret
n_threads: 2
max_context: 512
temperature_default: 0.5
top_p_default: 0.9
max_new_tokens_default: 32
memory_limit_bytes: 1073741824
sandbox_enabled: true
use_gpu: false
gpu_layers: 10
)");
    const lisa::Config cfg = lisa::load_config(file.path());

    CHECK(cfg.model_alias == "my-model");
    CHECK(cfg.chat_template == "llama3");
    CHECK(cfg.listen_addr == "0.0.0.0");
    CHECK(cfg.listen_port == 9000);
    CHECK(cfg.api_key == "secret");
    CHECK(cfg.n_threads == 2);
    CHECK(cfg.max_context == 512);
    CHECK(cfg.temperature_default == 0.5f);
    CHECK(cfg.max_new_tokens_default == 32);
    CHECK(cfg.memory_limit_bytes == 1073741824ULL);
    CHECK(cfg.sandbox_enabled);
    CHECK_FALSE(cfg.use_gpu);
    CHECK(cfg.gpu_layers == 10);
}

TEST_CASE("load_config rejects invalid files", "[config]") {
    using Catch::Matchers::ContainsSubstring;

    SECTION("missing file") {
        CHECK_THROWS_WITH(lisa::load_config("/nonexistent/lisa.yaml"),
                          ContainsSubstring("cannot read config"));
    }
    SECTION("missing model_path") {
        const TempConfig file("listen_port: 8080\n");
        CHECK_THROWS_WITH(lisa::load_config(file.path()), ContainsSubstring("model_path"));
    }
    SECTION("port out of range") {
        const TempConfig file("model_path: m.gguf\nlisten_port: 70000\n");
        CHECK_THROWS_WITH(lisa::load_config(file.path()), ContainsSubstring("listen_port"));
    }
    SECTION("wrong value type") {
        const TempConfig file("model_path: m.gguf\nn_threads: lots\n");
        CHECK_THROWS_WITH(lisa::load_config(file.path()), ContainsSubstring("invalid value"));
    }
    SECTION("TLS without certificate") {
        const TempConfig file("model_path: m.gguf\nenable_tls: true\n");
        CHECK_THROWS_WITH(lisa::load_config(file.path()), ContainsSubstring("cert_file"));
    }
    SECTION("not a mapping") {
        const TempConfig file("- a\n- b\n");
        CHECK_THROWS_WITH(lisa::load_config(file.path()), ContainsSubstring("mapping"));
    }
}

TEST_CASE("load_config ignores obsolete tokenizer keys", "[config]") {
    const TempConfig file("model_path: m.gguf\nvocab_path: v.json\nmerges_path: m.txt\n");
    CHECK_NOTHROW(lisa::load_config(file.path()));
}

TEST_CASE("is_loopback_address", "[config]") {
    CHECK(lisa::is_loopback_address("127.0.0.1"));
    CHECK(lisa::is_loopback_address("127.1.2.3"));
    CHECK(lisa::is_loopback_address("localhost"));
    CHECK(lisa::is_loopback_address("::1"));
    CHECK_FALSE(lisa::is_loopback_address("0.0.0.0"));
    CHECK_FALSE(lisa::is_loopback_address("192.168.1.10"));
    CHECK_FALSE(lisa::is_loopback_address("::"));
}

TEST_CASE("constant_time_equals", "[config]") {
    CHECK(lisa::constant_time_equals("secret", "secret"));
    CHECK_FALSE(lisa::constant_time_equals("secreT", "secret"));
    CHECK_FALSE(lisa::constant_time_equals("secret-but-longer", "secret"));
    CHECK_FALSE(lisa::constant_time_equals("sec", "secret"));
    CHECK_FALSE(lisa::constant_time_equals("", "secret"));
    CHECK(lisa::constant_time_equals("", ""));
}

TEST_CASE("load_config: n_threads 0 means one per available CPU", "[config]") {
    const TempConfig file("model_path: m.gguf\nn_threads: 0\n");
    const lisa::Config cfg = lisa::load_config(file.path(), fake_env());
    CHECK(cfg.n_threads == lisa::available_cpus());
    CHECK(cfg.n_threads >= 1);
}

TEST_CASE("load_config rejects unknown keys", "[config]") {
    using Catch::Matchers::ContainsSubstring;
    const TempConfig file("model_path: m.gguf\nlisten_prot: 9000\n");
    CHECK_THROWS_WITH(lisa::load_config(file.path(), fake_env()),
                      ContainsSubstring("unknown key 'listen_prot'"));
}

TEST_CASE("load_config: environment and command line override the file", "[config]") {
    const TempConfig file("model_path: m.gguf\nlisten_port: 9000\nn_threads: 2\n");

    lisa::Config cfg =
        lisa::load_config(file.path(), fake_env({{"LISA_LISTEN_PORT", "9001"},
                                                 {"LISA_API_KEY", "with#hash: and colon"},
                                                 {"LISA_API_KEYS", "k1,,k2"},
                                                 {"LISA_SANDBOX_ENABLED", "false"}}));
    CHECK(cfg.listen_port == 9001);
    CHECK(cfg.n_threads == 2);
    CHECK(cfg.api_key == "with#hash: and colon");  // taken literally, not parsed as YAML
    CHECK(cfg.api_keys == std::vector<std::string>{"k1", "k2"});
    CHECK_FALSE(cfg.sandbox_enabled);

    lisa::ConfigOverrides cli;
    cli.listen_port = 9002;
    cli.model_path = "other.gguf";
    cli.listen_addr = "0.0.0.0";
    cfg = lisa::load_config(file.path(), fake_env({{"LISA_LISTEN_PORT", "9001"}}), cli);
    CHECK(cfg.listen_port == 9002);
    CHECK(cfg.model_path == "other.gguf");
    CHECK(cfg.listen_addr == "0.0.0.0");
}

TEST_CASE("load_config without a file", "[config]") {
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(lisa::load_config("", fake_env()), ContainsSubstring("model_path"));
    const lisa::Config cfg = lisa::load_config("", fake_env({{"LISA_MODEL_PATH", "m.gguf"}}));
    CHECK(cfg.model_path == "m.gguf");
    CHECK(cfg.listen_port == 8080);

    const TempConfig empty("");
    lisa::ConfigOverrides cli;
    cli.model_path = "m.gguf";
    CHECK_NOTHROW(lisa::load_config(empty.path(), fake_env(), cli));
}

TEST_CASE("load_config: invalid environment values name the variable", "[config]") {
    using Catch::Matchers::ContainsSubstring;
    const TempConfig file("model_path: m.gguf\n");
    CHECK_THROWS_WITH(lisa::load_config(file.path(), fake_env({{"LISA_N_THREADS", "many"}})),
                      ContainsSubstring("LISA_N_THREADS"));
}

TEST_CASE("load_config reads api_keys_file", "[config]") {
    const std::string digest(64, 'a');
    const TempConfig keys("# team keys\n\n  first-key  \nsha256:" + digest + "\n");
    const TempConfig file("model_path: m.gguf\napi_keys: [inline]\napi_keys_file: " + keys.path() +
                          "\n");
    const lisa::Config cfg = lisa::load_config(file.path(), fake_env());
    CHECK(cfg.api_keys == std::vector<std::string>{"inline", "first-key", "sha256:" + digest});

    using Catch::Matchers::ContainsSubstring;
    const TempConfig missing("model_path: m.gguf\napi_keys_file: /nonexistent/keys\n");
    CHECK_THROWS_WITH(lisa::load_config(missing.path(), fake_env()),
                      ContainsSubstring("api_keys_file"));
}

TEST_CASE("load_config validates the P3 settings", "[config]") {
    using Catch::Matchers::ContainsSubstring;
    const auto rejects = [](const std::string& extra, const std::string& message) {
        const TempConfig file("model_path: m.gguf\n" + extra);
        CHECK_THROWS_WITH(lisa::load_config(file.path(), fake_env()), ContainsSubstring(message));
    };
    rejects("api_keys: ['sha256:1234']\n", "sha256:");
    rejects("api_keys: ['']\n", "empty");
    rejects("tls_min_version: '1.1'\n", "tls_min_version");
    rejects("client_ca_file: ca.pem\n", "client_ca_file requires enable_tls");
    rejects("max_queued_requests: -1\n", "max_queued_requests");
    rejects("rate_limit_burst: 0\n", "rate_limit_burst");
    rejects("read_timeout_seconds: 0\n", "read_timeout_seconds");
    rejects("max_prompt_tokens: -5\n", "max_prompt_tokens");
}

TEST_CASE("exposure_problem refuses weak keys off loopback", "[config]") {
    lisa::Config cfg;
    cfg.api_key = "changeme";
    CHECK(lisa::exposure_problem(cfg).empty());  // loopback: fine

    cfg.listen_addr = "0.0.0.0";
    CHECK_FALSE(lisa::exposure_problem(cfg).empty());
    cfg.api_key = "short";
    CHECK_FALSE(lisa::exposure_problem(cfg).empty());
    cfg.api_key = "a-long-random-key-0123";
    CHECK(lisa::exposure_problem(cfg).empty());
    cfg.api_keys = {"weak"};
    CHECK_FALSE(lisa::exposure_problem(cfg).empty());
    cfg.api_keys = {"sha256:" + std::string(64, 'f')};
    CHECK(lisa::exposure_problem(cfg).empty());
}
