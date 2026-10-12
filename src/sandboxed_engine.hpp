#pragma once

#include "engine.hpp"
#include "ipc.hpp"
#include "model_file.hpp"
#include "utils.hpp"

#include <sys/types.h>
#include <atomic>
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
    [[nodiscard]] EngineStatus status() const noexcept override;

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
    GenerationResult generate_locked(const std::vector<Token>& prompt_tokens,
                                     const GenerationParams& params, const PieceCallback& on_piece);
    [[noreturn]] void worker_failed(const ipc::IpcError& error);

    // Runs `decode` on messages from the worker. The worker is untrusted: a well-framed
    // message of the wrong shape is a protocol violation, so the worker is killed and
    // restarted rather than leaving the channel out of step.
    template <typename Decode>
    auto decode_worker_reply(Decode&& decode) -> decltype(decode()) {
        try {
            return decode();
        } catch (const json::exception& e) {
            worker_failed(
                ipc::IpcError(std::string("malformed message from the worker: ") + e.what(),
                              /*peer_closed=*/false));
        }
    }
    WorkerExit reap_worker(bool kill_first);

    Config cfg_;
    Options options_;
    ModelFile model_;
    std::mutex mutex_;  // one request at a time on the channel
    ipc::Channel channel_;
    pid_t pid_ = -1;
    std::deque<std::chrono::steady_clock::time_point> restarts_;

    // Lock-free status for /readyz and /metrics (mutex_ is held during generation).
    std::atomic<bool> worker_up_{false};
    std::atomic<std::chrono::steady_clock::rep> restart_blocked_until_{0};
    std::atomic<std::uint64_t> worker_deaths_{0};
    std::atomic<std::uint64_t> worker_restarts_{0};

    mutable std::mutex info_mutex_;
    ModelInfo info_;
    int landlock_abi_ = 0;
};

}  // namespace lisa
