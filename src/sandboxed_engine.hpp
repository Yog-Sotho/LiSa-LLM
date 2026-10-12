#pragma once

#include "engine.hpp"
#include "ipc.hpp"
#include "model_file.hpp"
#include "utils.hpp"

#include <sys/types.h>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace lisa {

// Engine whose model lives in a separate, sandboxed worker process (see sandbox.hpp). The
// broker side never parses the model file: it opens it, optionally checks its SHA-256 and
// passes the descriptor to the worker. A crashed worker is restarted on the next request,
// at most `max_restarts` times per `restart_window`.
class SandboxedEngine final : public Engine {
public:
    struct Options {
        std::string worker_exe = "/proc/self/exe";  // binary that handles kWorkerFlag
        int max_restarts = 3;
        std::chrono::seconds restart_window{60};
    };

    // Starts the worker and waits until the model is loaded. Throws std::runtime_error.
    SandboxedEngine(const Config& cfg, Options options);
    explicit SandboxedEngine(const Config& cfg) : SandboxedEngine(cfg, Options{}) {}
    ~SandboxedEngine() override;

    SandboxedEngine(const SandboxedEngine&) = delete;
    SandboxedEngine& operator=(const SandboxedEngine&) = delete;
    SandboxedEngine(SandboxedEngine&&) = delete;
    SandboxedEngine& operator=(SandboxedEngine&&) = delete;

    [[nodiscard]] std::vector<Token> tokenize(std::string_view text) override;
    [[nodiscard]] std::vector<Token> tokenize_chat(
        const std::vector<ChatMessage>& messages) override;
    GenerationResult generate(const std::vector<Token>& prompt_tokens,
                              const GenerationParams& params,
                              const PieceCallback& on_piece) override;
    [[nodiscard]] ModelInfo info() const override;

    // Test hook: runs a probe inside the sandbox. Returns the worker's answer, or
    // {"worker_died": ..., "signal": N, "exit_code": N} when the probe killed it.
    [[nodiscard]] json probe(const json& request);

    [[nodiscard]] pid_t worker_pid() const;
    [[nodiscard]] int landlock_abi() const;

private:
    struct WorkerExit {
        std::string description;
        int signal = 0;      // signal that killed the worker, or 0
        int exit_code = -1;  // exit code when it exited normally
    };

    void start_worker();
    void ensure_worker();
    json call(const json& request);
    [[noreturn]] void worker_failed(const ipc::IpcError& error);
    WorkerExit reap_worker(bool kill_first);

    Config cfg_;
    Options options_;
    ModelFile model_;
    std::mutex mutex_;  // one request at a time on the channel
    ipc::Channel channel_;
    pid_t pid_ = -1;
    std::deque<std::chrono::steady_clock::time_point> restarts_;

    mutable std::mutex info_mutex_;
    ModelInfo info_;
    int landlock_abi_ = 0;
};

}  // namespace lisa
