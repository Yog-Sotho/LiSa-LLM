#include "inference.hpp"
#include "sandbox.hpp"
#include "server.hpp"
#include "utils.hpp"

#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <ctime>
#include <exception>
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
constexpr std::string_view kPlaceholderApiKey = "changeme";
constexpr long kSignalPollNanos = 200'000'000;  // 200 ms

struct CliOptions {
    std::string config_path{kDefaultConfigPath};
};

void print_usage(std::ostream& out) {
    out << "Usage: lisa [--config <path>]\n"
           "\n"
           "Options:\n"
           "  -c, --config <path>  YAML configuration file (default: config.yaml)\n"
           "  -v, --version        Print the version and exit\n"
           "  -h, --help           Show this help and exit\n";
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
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            options.config_path = argv[++i];
            continue;
        }
        std::cerr << "lisa: unknown or incomplete argument '" << arg << "'\n";
        print_usage(std::cerr);
        exit_code = 2;
        return std::nullopt;
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

int run_server(const lisa::Config& cfg) {
    std::unique_ptr<lisa::InferenceEngine> engine;
    try {
        lisa::log_info("Loading model", {{"path", cfg.model_path}});
        engine = std::make_unique<lisa::InferenceEngine>(cfg);
    } catch (const std::exception& e) {
        lisa::log_error("Model loading failed", {{"error", e.what()}});
        return 1;
    }
    const lisa::ModelInfo& info = engine->info();
    lisa::log_info("Model loaded", {{"model", info.description},
                                    {"chat_template", info.chat_template},
                                    {"n_layer", info.n_layer},
                                    {"n_params", info.n_params},
                                    {"n_ctx", info.n_ctx},
                                    {"n_ctx_train", info.n_ctx_train},
                                    {"gpu", info.gpu_offload}});

    try {
        lisa::HttpServer server(*engine, cfg);
        const std::jthread signal_thread([&server](const std::stop_token& stop) {
            if (const int sig = wait_for_signal(stop); sig != 0) {
                lisa::log_info("Shutting down", {{"signal", sig}});
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

// Experimental: forks a sandboxed child that runs the server (see sandbox.hpp).
int run_sandboxed(const lisa::Config& cfg) {
    const pid_t child = fork();
    if (child < 0) {
        lisa::log_error("fork failed");
        return 1;
    }
    if (child == 0) {
        if (lisa::sandbox_init(cfg) != 0) {
            _exit(1);
        }
        _exit(run_server(cfg));
    }

    const sigset_t set = termination_signals();
    const timespec timeout{0, kSignalPollNanos};
    while (true) {
        int status = 0;
        if (waitpid(child, &status, WNOHANG) == child) {
            if (WIFSIGNALED(status)) {
                lisa::log_error("sandboxed server killed", {{"signal", WTERMSIG(status)}});
                return 1;
            }
            const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
            if (code != 0) {
                lisa::log_error("sandboxed server exited", {{"code", code}});
            }
            return code;
        }
        if (sigtimedwait(&set, nullptr, &timeout) > 0) {
            kill(child, SIGTERM);
            waitpid(child, nullptr, 0);
            return 0;
        }
    }
}

int run(int argc, char* argv[]) {
    int exit_code = 0;
    const std::optional<CliOptions> options = parse_cli(argc, argv, exit_code);
    if (!options) {
        return exit_code;
    }

    lisa::Config cfg;
    try {
        cfg = lisa::load_config(options->config_path);
    } catch (const std::exception& e) {
        lisa::log_error("Invalid configuration", {{"error", e.what()}});
        return 1;
    }

    if (cfg.api_key == kPlaceholderApiKey && !lisa::is_loopback_address(cfg.listen_addr)) {
        lisa::log_error(
            "Refusing to listen on a non-loopback address with the placeholder "
            "api_key 'changeme'; set a strong api_key",
            {{"listen_addr", cfg.listen_addr}});
        return 1;
    }
    if (cfg.api_key.empty() && !lisa::is_loopback_address(cfg.listen_addr)) {
        lisa::log_warn("API key authentication is disabled on a non-loopback address",
                       {{"listen_addr", cfg.listen_addr}});
    }

    // Block termination signals before any thread starts; a dedicated thread waits for them.
    const sigset_t set = termination_signals();
    pthread_sigmask(SIG_BLOCK, &set, nullptr);

    lisa::log_info("LiSa-LLM starting",
                   {{"version", lisa::kVersion}, {"sandbox", cfg.sandbox_enabled}});
    if (cfg.sandbox_enabled) {
        if (!lisa::sandbox_available()) {
            lisa::log_error(
                "sandbox_enabled is set but this binary was built with LISA_SANDBOX=OFF");
            return 1;
        }
        lisa::log_warn(
            "The sandbox is experimental and does not isolate the server yet "
            "(roadmap P2)");
        return run_sandboxed(cfg);
    }
    return run_server(cfg);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "lisa: fatal error: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "lisa: fatal error\n";
    }
    return 1;
}
