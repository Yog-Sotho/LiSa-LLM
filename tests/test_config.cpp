#include "utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

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

}  // namespace

TEST_CASE("load_config applies defaults for optional keys", "[config]") {
    const TempConfig file("model_path: ./m.gguf\n");
    const lisa::Config cfg = lisa::load_config(file.path());

    CHECK(cfg.model_path == "./m.gguf");
    CHECK(cfg.listen_addr == "127.0.0.1");
    CHECK(cfg.listen_port == 8080);
    CHECK_FALSE(cfg.enable_tls);
    CHECK_FALSE(cfg.sandbox_enabled);
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
