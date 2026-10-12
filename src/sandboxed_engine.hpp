#pragma once

#include "engine.hpp"
#include "ipc.hpp"
#include "model_file.hpp"
#include "utils.hpp"

#include <sys/types.h>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lisa {

// Engine whose model lives in a separate, sandboxed worker process (see sandbox.hpp). The
// broker side never parses the model file: it opens it, optionally checks its SHA-256 and
// passes the descriptor to the worker. Requests are multiplexed over one connection (each
// message carries an id), so the worker batches concurrent generations. A crashed worker
// is restarted on the next request, at most `max_restarts` times per `restart_window`.
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

    // All thread-safe; concurrent generations run in the worker's batch.
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
    struct Worker;   // one worker process and its connection
    struct Mailbox;  // replies to one request
    class MailboxCloser;

    [[nodiscard]] std::shared_ptr<Worker> start_worker();
    [[nodiscard]] std::shared_ptr<Worker> acquire_worker();
    void read_loop(const std::shared_ptr<Worker>& worker);
    WorkerExit reap(Worker& worker, bool kill_first) const;
    void kill_worker(Worker& worker, const std::string& reason);

    // Sends `request` with a fresh id and passes its single reply to `decode` (an error
    // reply is thrown as RequestError/runtime_error). Throws std::runtime_error if the
    // worker dies.
    void call(json request, const std::function<void(const json&)>& decode);

    // Decodes a worker message. The worker is untrusted: a well-framed message of the
    // wrong shape is a protocol violation, so the worker is killed (and later restarted).
    template <typename Decode>
    auto decode_reply(Worker& worker, Decode&& decode) -> decltype(decode());

    Config cfg_;
    Options options_;
    ModelFile model_;

    std::mutex lifecycle_mutex_;  // guards worker_, restarts_
    std::shared_ptr<Worker> worker_;
    std::deque<std::chrono::steady_clock::time_point> restarts_;

    // Lock-free status for /readyz and /metrics.
    std::atomic<bool> worker_up_{false};
    std::atomic<std::chrono::steady_clock::rep> restart_blocked_until_{0};
    std::atomic<std::uint64_t> worker_deaths_{0};
    std::atomic<std::uint64_t> worker_restarts_{0};
    std::atomic<pid_t> worker_pid_{-1};

    mutable std::mutex info_mutex_;
    ModelInfo info_;
    int landlock_abi_ = 0;
};

}  // namespace lisa
