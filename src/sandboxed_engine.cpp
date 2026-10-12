#include "sandboxed_engine.hpp"

#include "protocol.hpp"
#include "sandbox.hpp"
#include "worker.hpp"

#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace lisa {

namespace {

constexpr int kDescriptorFloor = 10;  // keep spawn sources clear of fds 3 and 4
constexpr auto kShutdownGrace = std::chrono::seconds(2);
constexpr auto kReapPoll = std::chrono::milliseconds(20);

// Only sanitizer settings are passed through; the worker gets no other environment.
std::vector<std::string> worker_environment() {
    std::vector<std::string> env;
    bool asan_seen = false;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const std::string_view entry(*e);
        if (entry.starts_with("ASAN_OPTIONS=")) {
            // LeakSanitizer needs /proc and fork(), neither of which the sandbox allows.
            env.emplace_back(std::string(entry) + ":detect_leaks=0");
            asan_seen = true;
        } else if (entry.starts_with("TSAN_OPTIONS=") || entry.starts_with("UBSAN_OPTIONS=") ||
                   entry.starts_with("LSAN_OPTIONS=")) {
            env.emplace_back(entry);
        }
    }
    if (!asan_seen) {
        env.emplace_back("ASAN_OPTIONS=detect_leaks=0");
    }
    return env;
}

std::string signal_name(int sig) {
    const char* name = ::sigabbrev_np(sig);
    return name != nullptr ? std::string("SIG") + name : "signal " + std::to_string(sig);
}

int duplicate_high(int fd) {
    const int copy = ::fcntl(fd, F_DUPFD_CLOEXEC, kDescriptorFloor);
    if (copy < 0) {
        throw std::runtime_error(std::string("sandbox: fcntl: ") + std::strerror(errno));
    }
    return copy;
}

}  // namespace

struct SandboxedEngine::Mailbox {
    std::condition_variable changed;
    std::deque<json> messages;
};

struct SandboxedEngine::Worker {
    pid_t pid = -1;
    // Opened once and closed by the destructor only (after the reader thread has joined),
    // so concurrent send()/shutdown() calls always see a valid descriptor.
    ipc::Channel channel;
    std::mutex send_mutex;  // one writer at a time
    std::atomic<std::uint64_t> next_id{1};
    std::thread reader;

    std::mutex mutex;  // guards the members below
    std::unordered_map<std::uint64_t, std::shared_ptr<Mailbox>> mailboxes;
    bool dead = false;     // no more replies will arrive
    bool reaped = false;   // pid has been waited for (must not be signalled any more)
    bool counted = false;  // the death was logged and counted
    std::string death;     // why it died, for errors
    WorkerExit exit;       // set before `dead`

    // Sends one request. Throws std::runtime_error when the worker is gone.
    void send(const json& message) {
        const std::lock_guard<std::mutex> lock(send_mutex);
        try {
            channel.send(message);
        } catch (const ipc::IpcError&) {
            // The reader thread notices the same failure and reports the death.
            throw std::runtime_error("inference worker died (connection lost)");
        }
    }

    std::shared_ptr<Mailbox> open(std::uint64_t id) {
        auto box = std::make_shared<Mailbox>();
        const std::lock_guard<std::mutex> lock(mutex);
        if (dead) {
            throw std::runtime_error("inference worker died (" + death + ")");
        }
        mailboxes.emplace(id, box);
        return box;
    }

    void close(std::uint64_t id) {
        const std::lock_guard<std::mutex> lock(mutex);
        mailboxes.erase(id);
    }

    // Next message for `box`; throws std::runtime_error once the worker is dead.
    json next(const std::shared_ptr<Mailbox>& box) {
        std::unique_lock<std::mutex> lock(mutex);
        box->changed.wait(lock, [&] { return !box->messages.empty() || dead; });
        if (box->messages.empty()) {
            throw std::runtime_error("inference worker died (" + death + ")");
        }
        json message = std::move(box->messages.front());
        box->messages.pop_front();
        return message;
    }
};

// Removes a request's mailbox when the caller is done with it, however it leaves.
class SandboxedEngine::MailboxCloser {
public:
    MailboxCloser(Worker& worker, std::uint64_t id) : worker_(worker), id_(id) {}
    MailboxCloser(const MailboxCloser&) = delete;
    MailboxCloser& operator=(const MailboxCloser&) = delete;
    MailboxCloser(MailboxCloser&&) = delete;
    MailboxCloser& operator=(MailboxCloser&&) = delete;
    ~MailboxCloser() { worker_.close(id_); }

private:
    Worker& worker_;
    std::uint64_t id_;
};

SandboxedEngine::SandboxedEngine(const Config& cfg, Options options)
    : cfg_(cfg), options_(std::move(options)), model_(cfg.model_path) {
    if (!sandbox_available()) {
        throw std::runtime_error(
            "sandbox_enabled is set but this binary was built with LISA_SANDBOX=OFF");
    }
#ifdef LISA_CUDA_BUILD
    if (cfg_.use_gpu) {
        throw std::runtime_error(
            "GPU offload inside the sandbox is not supported yet; set use_gpu: false or "
            "sandbox_enabled: false");
    }
#endif
    if (!cfg_.model_sha256.empty()) {
        model_.verify_sha256(cfg_.model_sha256);
    }
    worker_ = start_worker();
}

SandboxedEngine::~SandboxedEngine() {
    const std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (worker_) {
        {
            const std::lock_guard<std::mutex> worker_lock(worker_->mutex);
            worker_->counted = true;  // an orderly stop, not a death
        }
        worker_->channel.shutdown();  // the worker exits when the broker hangs up
        if (worker_->reader.joinable()) {
            worker_->reader.join();
        }
    }
}

std::shared_ptr<SandboxedEngine::Worker> SandboxedEngine::start_worker() {
    std::array<int, 2> sv{};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv.data()) != 0) {
        throw std::runtime_error(std::string("sandbox: socketpair: ") + std::strerror(errno));
    }
    ipc::Channel broker_end(sv[0]);
    const int worker_end = duplicate_high(sv[1]);
    ::close(sv[1]);
    const int model_copy = duplicate_high(model_.fd());

    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_adddup2(&actions, worker_end, kWorkerChannelFd);
    posix_spawn_file_actions_adddup2(&actions, model_copy, kWorkerModelFd);
    posix_spawn_file_actions_addclosefrom_np(&actions, kWorkerModelFd + 1);

    sigset_t no_signals;
    sigemptyset(&no_signals);
    sigset_t defaults;
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    sigaddset(&defaults, SIGINT);
    sigaddset(&defaults, SIGTERM);
    posix_spawnattr_setsigmask(&attr, &no_signals);
    posix_spawnattr_setsigdefault(&attr, &defaults);
    posix_spawnattr_setpgroup(&attr, 0);  // Ctrl-C goes to the broker, which stops the worker
    posix_spawnattr_setflags(
        &attr, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETPGROUP);

    std::string exe = options_.worker_exe;
    std::string flag(kWorkerFlag);
    std::vector<char*> argv = {exe.data(), flag.data(), nullptr};
    std::vector<std::string> env_strings = worker_environment();
    std::vector<char*> envp(env_strings.size() + 1, nullptr);
    // posix_spawn takes char* const[], so the strings must be reachable as non-const.
    // cppcheck-suppress constParameterReference
    const auto as_mutable = [](std::string& entry) { return entry.data(); };
    std::transform(env_strings.begin(), env_strings.end(), envp.begin(), as_mutable);

    pid_t pid = -1;
    const int spawned = ::posix_spawn(&pid, exe.c_str(), &actions, &attr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    ::close(worker_end);
    ::close(model_copy);
    if (spawned != 0) {
        throw std::runtime_error("sandbox: cannot start worker '" + exe +
                                 "': " + std::strerror(spawned));
    }
    auto worker = std::make_shared<Worker>();
    worker->pid = pid;
    worker->channel = std::move(broker_end);

    json reply;
    try {
        worker->channel.send({{"op", "init"}, {"settings", WorkerSettings::from_config(cfg_)}});
        reply = worker->channel.receive();
    } catch (const ipc::IpcError& e) {
        const WorkerExit exit = reap(*worker, !e.peer_closed());
        std::string message = "sandboxed worker failed to start (" + exit.description + ")";
        if (exit.signal != SIGSYS && cfg_.memory_limit_bytes > 0) {
            // Out-of-memory inside libllama usually ends in an assertion or a crash.
            message +=
                "; if the log shows an allocation failure, raise memory_limit_bytes "
                "or lower parallel_slots or max_context";
        }
        throw std::runtime_error(message);
    }
    try {
        if (reply.contains("fatal")) {
            worker->channel.close();
            (void)reap(*worker, false);
            throw std::runtime_error(reply["fatal"].get<std::string>());
        }
        const std::lock_guard<std::mutex> lock(info_mutex_);
        info_ = reply.at("ready").get<ModelInfo>();
        landlock_abi_ = reply.value("landlock_abi", 0);
    } catch (const json::exception& e) {
        (void)reap(*worker, true);
        throw std::runtime_error(std::string("sandboxed worker sent a malformed reply: ") +
                                 e.what());
    }
    worker_pid_ = worker->pid;
    worker_up_ = true;
    log_info("Sandboxed worker ready", {{"pid", worker->pid}, {"landlock_abi", landlock_abi_}});
    worker->reader = std::thread([this, worker] { read_loop(worker); });
    return worker;
}

void SandboxedEngine::read_loop(const std::shared_ptr<Worker>& worker) {
    std::string reason;
    bool kill_first = false;
    while (true) {
        json message;
        try {
            message = worker->channel.receive();
        } catch (const ipc::IpcError& e) {
            reason = e.what();
            kill_first = !e.peer_closed();
            break;
        }
        const json* id = message.is_object() ? &message["id"] : nullptr;
        if (id == nullptr || !id->is_number_unsigned()) {
            reason = "malformed message from the worker: no request id";
            kill_first = true;
            break;
        }
        const std::lock_guard<std::mutex> lock(worker->mutex);
        if (const auto it = worker->mailboxes.find(id->get<std::uint64_t>());
            it != worker->mailboxes.end()) {
            it->second->messages.push_back(std::move(message));
            it->second->changed.notify_all();
        }  // else: for a request whose caller has gone (e.g. its HTTP thread threw)
    }

    worker_up_ = false;
    const WorkerExit exit = reap(*worker, kill_first);
    bool report = false;
    {
        const std::lock_guard<std::mutex> lock(worker->mutex);
        if (!worker->counted) {
            worker->counted = true;
            report = true;
            if (worker->death.empty()) {
                worker->death = exit.description;
            }
        }
        worker->exit = exit;
        worker->dead = true;
        for (const auto& [request_id, box] : worker->mailboxes) {
            box->changed.notify_all();
        }
    }
    // The descriptor stays open until the Worker is destroyed: other threads may still
    // call channel.shutdown(), and a closed (reused) descriptor number must never be hit.
    if (report) {
        ++worker_deaths_;
        log_error("Inference worker died", {{"status", exit.description}, {"error", reason}});
    }
}

void SandboxedEngine::kill_worker(Worker& worker, const std::string& reason) {
    bool report = false;
    {
        const std::lock_guard<std::mutex> lock(worker.mutex);
        if (!worker.counted) {
            worker.counted = true;
            report = true;
            worker.death = reason;
        }
        if (!worker.reaped) {
            ::kill(worker.pid, SIGKILL);  // the reader thread reaps it and wakes everyone
        }
    }
    worker_up_ = false;
    if (report) {
        ++worker_deaths_;
        log_error("Inference worker died", {{"status", "killed by the broker"}, {"error", reason}});
    }
}

SandboxedEngine::WorkerExit SandboxedEngine::reap(Worker& worker, bool kill_first) const {
    WorkerExit exit;
    if (kill_first) {
        const std::lock_guard<std::mutex> lock(worker.mutex);
        ::kill(worker.pid, SIGKILL);
    }
    int wait_status = 0;
    const auto deadline = std::chrono::steady_clock::now() + kShutdownGrace;
    while (true) {
        {
            // Under the lock, so kill_worker() never signals a reaped (reusable) pid.
            const std::lock_guard<std::mutex> lock(worker.mutex);
            const pid_t r = ::waitpid(worker.pid, &wait_status, WNOHANG);
            if (r == worker.pid || (r < 0 && errno != EINTR)) {
                worker.reaped = true;
                break;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                ::kill(worker.pid, SIGKILL);
                ::waitpid(worker.pid, &wait_status, 0);
                worker.reaped = true;
                break;
            }
        }
        std::this_thread::sleep_for(kReapPoll);
    }

    // The reaper reports a killed worker as exit code 128 + signal.
    if (WIFSIGNALED(wait_status)) {
        exit.signal = WTERMSIG(wait_status);
    } else if (WIFEXITED(wait_status)) {
        const int code = WEXITSTATUS(wait_status);
        if (code > 128) {
            exit.signal = code - 128;
        } else {
            exit.exit_code = code;
        }
    }
    if (exit.signal != 0) {
        exit.description = "killed by " + signal_name(exit.signal);
        if (exit.signal == SIGSYS) {
            exit.description += " (blocked system call)";
        }
    } else {
        exit.description = "exited with code " + std::to_string(exit.exit_code);
    }
    if (!cfg_.sandbox_cgroup.empty()) {
        ::rmdir((cfg_.sandbox_cgroup + "/lisa-worker-" + std::to_string(worker.pid)).c_str());
    }
    return exit;
}

std::shared_ptr<SandboxedEngine::Worker> SandboxedEngine::acquire_worker() {
    const std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (worker_) {
        const std::lock_guard<std::mutex> worker_lock(worker_->mutex);
        if (!worker_->dead && !worker_->counted) {
            return worker_;
        }
    }
    if (worker_) {
        // Its reader thread is finishing (or done) reaping; wait for it.
        if (worker_->reader.joinable()) {
            worker_->reader.join();
        }
        worker_.reset();
    }
    const auto now = std::chrono::steady_clock::now();
    while (!restarts_.empty() && now - restarts_.front() > options_.restart_window) {
        restarts_.pop_front();
    }
    if (static_cast<int>(restarts_.size()) >= options_.max_restarts) {
        restart_blocked_until_ =
            (restarts_.front() + options_.restart_window).time_since_epoch().count();
        throw std::runtime_error("inference worker keeps crashing; not restarting it");
    }
    restarts_.push_back(now);
    ++worker_restarts_;
    log_warn("Restarting the inference worker");
    worker_ = start_worker();
    return worker_;
}

template <typename Decode>
auto SandboxedEngine::decode_reply(Worker& worker, Decode&& decode) -> decltype(decode()) {
    try {
        return decode();
    } catch (const json::exception& e) {
        const std::string reason = std::string("malformed message from the worker: ") + e.what();
        kill_worker(worker, reason);
        throw std::runtime_error("inference worker died (" + reason + ")");
    }
}

void SandboxedEngine::call(json request, const std::function<void(const json&)>& decode) {
    const std::shared_ptr<Worker> worker = acquire_worker();
    const std::uint64_t id = worker->next_id++;
    request["id"] = id;
    const std::shared_ptr<Mailbox> box = worker->open(id);
    const MailboxCloser closer(*worker, id);
    worker->send(request);
    const json reply = worker->next(box);
    decode_reply(*worker, [&] {
        if (reply.contains("error")) {
            throw_error_reply(reply);
        }
        decode(reply);
    });
}

std::vector<Token> SandboxedEngine::tokenize(std::string_view text) {
    std::vector<Token> tokens;
    call({{"op", "tokenize"}, {"text", std::string(text)}},
         [&](const json& reply) { tokens = reply.at("tokens").get<std::vector<Token>>(); });
    return tokens;
}

std::vector<Token> SandboxedEngine::tokenize_chat(const std::vector<ChatMessage>& messages) {
    std::vector<Token> tokens;
    call({{"op", "tokenize_chat"}, {"messages", messages}},
         [&](const json& reply) { tokens = reply.at("tokens").get<std::vector<Token>>(); });
    return tokens;
}

GenerationResult SandboxedEngine::generate(const std::vector<Token>& prompt_tokens,
                                           const GenerationParams& params,
                                           const PieceCallback& on_piece) {
    const std::shared_ptr<Worker> worker = acquire_worker();
    const std::uint64_t id = worker->next_id++;
    const std::shared_ptr<Mailbox> box = worker->open(id);
    const MailboxCloser closer(*worker, id);
    worker->send({{"op", "generate"}, {"id", id}, {"tokens", prompt_tokens}, {"params", params}});

    bool cancelled = false;
    const auto cancel = [&] {
        if (!cancelled) {
            cancelled = true;
            worker->send({{"op", "cancel"}, {"id", id}});
        }
    };
    while (true) {
        const json message = worker->next(box);
        bool keep_going = true;
        std::optional<GenerationResult> done;
        decode_reply(*worker, [&] {
            if (message.contains("piece")) {
                const auto& piece = message.at("piece").get_ref<const std::string&>();
                if (!cancelled) {
                    keep_going = on_piece(piece);
                }
            } else if (message.contains("done")) {
                done = message.at("done").get<GenerationResult>();
            } else if (message.contains("error")) {
                throw_error_reply(message);
            } else {
                throw json::other_error::create(501, "unexpected message", nullptr);
            }
        });
        if (done) {
            if (cancelled) {
                done->reason = StopReason::cancelled;
            }
            return *done;
        }
        if (!keep_going) {
            cancel();  // keep reading until the worker confirms with "done"
        }
    }
}

ModelInfo SandboxedEngine::info() const {
    const std::lock_guard<std::mutex> lock(info_mutex_);
    return info_;
}

json SandboxedEngine::probe(const json& request) {
    const std::shared_ptr<Worker> worker = acquire_worker();
    json message = request;
    message["op"] = "probe";
    try {
        json answer;
        call(message, [&](const json& reply) { answer = reply.at("probe"); });
        return answer;
    } catch (const std::runtime_error&) {
        const std::lock_guard<std::mutex> lock(worker->mutex);
        if (!worker->dead) {
            throw;
        }
        return json{{"worker_died", worker->exit.description},
                    {"signal", worker->exit.signal},
                    {"exit_code", worker->exit.exit_code}};
    }
}

EngineStatus SandboxedEngine::status() const noexcept {
    EngineStatus status;
    status.ready = worker_up_ || std::chrono::steady_clock::now().time_since_epoch().count() >=
                                     restart_blocked_until_;
    status.worker_deaths = worker_deaths_;
    status.worker_restarts = worker_restarts_;
    return status;
}

pid_t SandboxedEngine::worker_pid() const {
    return worker_pid_;
}

int SandboxedEngine::landlock_abi() const {
    const std::lock_guard<std::mutex> lock(info_mutex_);
    return landlock_abi_;
}

}  // namespace lisa
