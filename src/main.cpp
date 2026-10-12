#include "auth.hpp"
#include "inference.hpp"
#include "model_file.hpp"
#include "sandboxed_engine.hpp"
#include "server.hpp"
#include "utils.hpp"
#include "worker.hpp"

#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#include <charconv>
#include <csignal>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iostream>
#include <lisa/version.hpp>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

namespace {

constexpr std::string_view kDefaultConfigPath = "config.yaml";
constexpr long kSignalPollNanos = 200'000'000;  // 200 ms
constexpr int kUsageError = 2;

struct CliOptions {
    std::string config_path;  // empty: config.yaml if it exists, else defaults only
    bool config_explicit = false;
    lisa::ConfigOverrides overrides;
    bool check_config = false;
};

void print_usage(std::ostream& out) {
    out << "Usage: lisa [options]\n"
           "\n"
           "Options:\n"
           "  -c, --config <path>  YAML configuration file (default: config.yaml if present)\n"
           "  -m, --model <path>   GGUF model file (overrides model_path)\n"
           "      --host <addr>    Address to listen on (overrides listen_addr)\n"
           "  -p, --port <port>    Port to listen on (overrides listen_port)\n"
           "      --check-config   Validate the configuration and the model file, then exit\n"
           "      --hash-key       Read an API key from stdin and print its sha256: entry\n"
           "  -v, --version        Print the version and exit\n"
           "  -h, --help           Show this help and exit\n"
           "\n"
           "Every config key can also be set with an environment variable named LISA_<KEY>,\n"
           "e.g. LISA_LISTEN_PORT=9000 or LISA_API_KEYS=key1,key2.\n"
           "Precedence: command line > environment > config file > defaults.\n";
}

std::optional<int> parse_port(std::string_view text) {
    int port = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), port);
    if (error != std::errc() || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return port;
}

// Prints "sha256:<digest>" for the key on the first line of stdin.
int hash_key() {
    std::string key;
    std::getline(std::cin, key);
    if (!key.empty() && key.back() == '\r') {
        key.pop_back();
    }
    if (key.empty()) {
        std::cerr << "lisa: --hash-key expects the key on standard input\n";
        return 1;
    }
    std::cout << "sha256:" << lisa::sha256_hex(key) << '\n';
    return 0;
}

// Returns the parsed options, or an exit code when the process should stop.
std::optional<CliOptions> parse_cli(int argc, const char* const argv[], int& exit_code) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            exit_code = 0;
            return std::nullopt;
        }
        if (arg == "-v" || arg == "--version") {
            std::cout << "lisa " << lisa::kVersion << '\n';
            exit_code = 0;
            return std::nullopt;
        }
        if (arg == "--hash-key") {
            exit_code = hash_key();
            return std::nullopt;
        }
        if (arg == "--check-config") {
            options.check_config = true;
            continue;
        }
        const bool takes_value = arg == "-c" || arg == "--config" || arg == "-m" ||
                                 arg == "--model" || arg == "--host" || arg == "-p" ||
                                 arg == "--port";
        if (takes_value && i + 1 < argc) {
            const std::string_view next = argv[++i];
            if (arg == "-c" || arg == "--config") {
                options.config_path = std::string(next);
                options.config_explicit = true;
            } else if (arg == "-m" || arg == "--model") {
                options.overrides.model_path = std::string(next);
            } else if (arg == "--host") {
                options.overrides.listen_addr = std::string(next);
            } else {
                options.overrides.listen_port = parse_port(next);
                if (!options.overrides.listen_port) {
                    std::cerr << "lisa: invalid port '" << next << "'\n";
                    exit_code = kUsageError;
                    return std::nullopt;
                }
            }
            continue;
        }
        std::cerr << "lisa: unknown or incomplete argument '" << arg << "'\n";
        print_usage(std::cerr);
        exit_code = kUsageError;
        return std::nullopt;
    }
    if (!options.config_explicit && std::filesystem::exists(kDefaultConfigPath)) {
        options.config_path = std::string(kDefaultConfigPath);
    }
    return options;
}

sigset_t termination_signals() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    return set;
}

// Waits for SIGINT/SIGTERM (blocked in every thread) and returns the signal, or 0 when
// `stop` is requested first.
int wait_for_signal(const std::stop_token& stop) {
    const sigset_t set = termination_signals();
    const timespec timeout{0, kSignalPollNanos};
    while (!stop.stop_requested()) {
        const int sig = sigtimedwait(&set, nullptr, &timeout);
        if (sig > 0) {
            return sig;
        }
    }
    return 0;
}

// The model runs in a sandboxed worker process unless the sandbox is disabled; then it
// is loaded in-process through the same verified descriptor.
struct EngineHandle {
    std::unique_ptr<lisa::ModelFile> model;  // in-process mode only
    std::unique_ptr<lisa::Engine> engine;
};

EngineHandle make_engine(const lisa::Config& cfg) {
    EngineHandle handle;
    if (cfg.sandbox_enabled) {
        handle.engine = std::make_unique<lisa::SandboxedEngine>(cfg);
        return handle;
    }
    lisa::log_warn("The sandbox is disabled: the model is parsed inside the server process");
    handle.model = std::make_unique<lisa::ModelFile>(cfg.model_path);
    if (!cfg.model_sha256.empty()) {
        handle.model->verify_sha256(cfg.model_sha256);
    }
    lisa::Config local = cfg;
    local.model_path = handle.model->proc_path();
    handle.engine = std::make_unique<lisa::InferenceEngine>(local);
    return handle;
}

// --check-config: the model file must be readable (and match model_sha256 if set).
int check_model(const lisa::Config& cfg) {
    try {
        const lisa::ModelFile model(cfg.model_path);
        if (!cfg.model_sha256.empty()) {
            model.verify_sha256(cfg.model_sha256);
        }
    } catch (const std::exception& e) {
        lisa::log_error("Invalid configuration", {{"error", e.what()}});
        return 1;
    }
    std::cout << "configuration OK\n";
    return 0;
}

int run_server(const lisa::Config& cfg) {
    EngineHandle handle;
    try {
        lisa::log_info("Loading model",
                       {{"path", cfg.model_path}, {"sandbox", cfg.sandbox_enabled}});
        handle = make_engine(cfg);
    } catch (const std::exception& e) {
        lisa::log_error("Model loading failed", {{"error", e.what()}});
        return 1;
    }
    lisa::Engine* engine = handle.engine.get();
    const lisa::ModelInfo info = engine->info();
    lisa::log_info("Model loaded", {{"model", info.description},
                                    {"chat_template", info.chat_template},
                                    {"n_layer", info.n_layer},
                                    {"n_params", info.n_params},
                                    {"n_ctx", info.n_ctx},
                                    {"n_ctx_train", info.n_ctx_train},
                                    {"gpu", info.gpu_offload}});

    try {
        lisa::HttpServer server(*engine, cfg);
        // First SIGINT/SIGTERM: drain gracefully. A second one: stop at once.
        std::jthread drain_thread;
        const std::jthread signal_thread([&server, &drain_thread](const std::stop_token& stop) {
            const int sig = wait_for_signal(stop);
            if (sig == 0) {
                return;
            }
            lisa::log_info("Shutting down", {{"signal", sig}});
            drain_thread = std::jthread([&server] { server.shutdown(); });
            if (const int again = wait_for_signal(stop); again != 0) {
                lisa::log_warn("Second signal: stopping without waiting", {{"signal", again}});
                server.stop();
            }
        });
        server.run();
    } catch (const std::exception& e) {
        lisa::log_error("Server error", {{"error", e.what()}});
        return 1;
    }
    return 0;
}

int run(int argc, char* argv[]) {
    int exit_code = 0;
    const std::optional<CliOptions> options = parse_cli(argc, argv, exit_code);
    if (!options) {
        return exit_code;
    }

    lisa::Config cfg;
    try {
        cfg = lisa::load_config(options->config_path, lisa::process_environment(),
                                options->overrides);
    } catch (const std::exception& e) {
        lisa::log_error("Invalid configuration", {{"error", e.what()}});
        return 1;
    }

    if (const std::string problem = lisa::exposure_problem(cfg); !problem.empty()) {
        lisa::log_error("Refusing to listen on a non-loopback address: " + problem +
                            "; use a random key of at least 16 characters",
                        {{"listen_addr", cfg.listen_addr}});
        return 1;
    }
    if (cfg.api_key.empty() && cfg.api_keys.empty() &&
        !lisa::is_loopback_address(cfg.listen_addr)) {
        lisa::log_warn("API key authentication is disabled on a non-loopback address",
                       {{"listen_addr", cfg.listen_addr}});
    }
    if (options->check_config) {
        return check_model(cfg);
    }

    // Block termination signals before any thread starts; a dedicated thread waits for them.
    const sigset_t set = termination_signals();
    pthread_sigmask(SIG_BLOCK, &set, nullptr);

    lisa::log_info("LiSa-LLM starting",
                   {{"version", lisa::kVersion}, {"sandbox", cfg.sandbox_enabled}});
    return run_server(cfg);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == lisa::kWorkerFlag) {
        return lisa::worker_main();
    }
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "lisa: fatal error: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "lisa: fatal error\n";
    }
    return 1;
}
