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
#include <csignal>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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
    start_worker();
}

SandboxedEngine::~SandboxedEngine() {
    if (pid_ > 0) {
        channel_.close();  // the worker exits when the broker hangs up
        (void)reap_worker(/*kill_first=*/false);
    }
}

void SandboxedEngine::start_worker() {
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
    pid_ = pid;
    channel_ = std::move(broker_end);

    json reply;
    try {
        channel_.send({{"op", "init"}, {"settings", WorkerSettings::from_config(cfg_)}});
        reply = channel_.receive();
    } catch (const ipc::IpcError& e) {
        const WorkerExit exit = reap_worker(!e.peer_closed());
        std::string message = "sandboxed worker failed to start (" + exit.description + ")";
        if (exit.signal != SIGSYS && cfg_.memory_limit_bytes > 0) {
            // Out-of-memory inside libllama usually ends in an assertion or a crash.
            message += "; if the log shows an allocation failure, raise memory_limit_bytes";
        }
        throw std::runtime_error(message);
    }
    if (reply.contains("fatal")) {
        channel_.close();
        (void)reap_worker(false);
        throw std::runtime_error(reply["fatal"].get<std::string>());
    }

    {
        const std::lock_guard<std::mutex> lock(info_mutex_);
        info_ = reply.at("ready").get<ModelInfo>();
        landlock_abi_ = reply.value("landlock_abi", 0);
    }
    log_info("Sandboxed worker ready", {{"pid", pid_}, {"landlock_abi", landlock_abi_}});
}

SandboxedEngine::WorkerExit SandboxedEngine::reap_worker(bool kill_first) {
    WorkerExit exit;
    if (pid_ <= 0) {
        exit.description = "not running";
        return exit;
    }
    channel_.close();
    if (kill_first) {
        ::kill(pid_, SIGKILL);
    }
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + kShutdownGrace;
    while (true) {
        const pid_t r = ::waitpid(pid_, &status, WNOHANG);
        if (r == pid_ || (r < 0 && errno != EINTR)) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
            break;
        }
        std::this_thread::sleep_for(kReapPoll);
    }

    // The reaper reports a killed worker as exit code 128 + signal.
    if (WIFSIGNALED(status)) {
        exit.signal = WTERMSIG(status);
    } else if (WIFEXITED(status)) {
        const int code = WEXITSTATUS(status);
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
        ::rmdir((cfg_.sandbox_cgroup + "/lisa-worker-" + std::to_string(pid_)).c_str());
    }
    pid_ = -1;
    return exit;
}

void SandboxedEngine::worker_failed(const ipc::IpcError& error) {
    const WorkerExit exit = reap_worker(/*kill_first=*/!error.peer_closed());
    log_error("Inference worker died", {{"status", exit.description}, {"error", error.what()}});
    throw std::runtime_error("inference worker died (" + exit.description + ")");
}

void SandboxedEngine::ensure_worker() {
    if (channel_.is_open()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    while (!restarts_.empty() && now - restarts_.front() > options_.restart_window) {
        restarts_.pop_front();
    }
    if (static_cast<int>(restarts_.size()) >= options_.max_restarts) {
        throw std::runtime_error("inference worker keeps crashing; not restarting it");
    }
    restarts_.push_back(now);
    log_warn("Restarting the inference worker");
    start_worker();
}

json SandboxedEngine::call(const json& request) {
    ensure_worker();
    json reply;
    try {
        channel_.send(request);
        reply = channel_.receive();
    } catch (const ipc::IpcError& e) {
        worker_failed(e);
    }
    if (reply.contains("error")) {
        throw_error_reply(reply);
    }
    return reply;
}

std::vector<Token> SandboxedEngine::tokenize(std::string_view text) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return call({{"op", "tokenize"}, {"text", std::string(text)}})
        .at("tokens")
        .get<std::vector<Token>>();
}

std::vector<Token> SandboxedEngine::tokenize_chat(const std::vector<ChatMessage>& messages) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return call({{"op", "tokenize_chat"}, {"messages", messages}})
        .at("tokens")
        .get<std::vector<Token>>();
}

GenerationResult SandboxedEngine::generate(const std::vector<Token>& prompt_tokens,
                                           const GenerationParams& params,
                                           const PieceCallback& on_piece) {
    const std::lock_guard<std::mutex> lock(mutex_);
    ensure_worker();
    try {
        bool cancelled = false;
        channel_.send({{"op", "generate"}, {"tokens", prompt_tokens}, {"params", params}});
        while (true) {
            json message = channel_.receive();
            if (message.contains("piece")) {
                if (!cancelled && !on_piece(message["piece"].get_ref<const std::string&>())) {
                    cancelled = true;
                    channel_.send({{"op", "cancel"}});
                }
            } else if (message.contains("done")) {
                auto result = message["done"].get<GenerationResult>();
                if (cancelled) {
                    result.reason = StopReason::cancelled;
                }
                return result;
            } else if (message.contains("error")) {
                throw_error_reply(message);
            } else {
                throw ipc::IpcError("unexpected message from worker", false);
            }
        }
    } catch (const ipc::IpcError& e) {
        worker_failed(e);
    }
}

ModelInfo SandboxedEngine::info() const {
    const std::lock_guard<std::mutex> lock(info_mutex_);
    return info_;
}

json SandboxedEngine::probe(const json& request) {
    const std::lock_guard<std::mutex> lock(mutex_);
    ensure_worker();
    try {
        json message = request;
        message["op"] = "probe";
        channel_.send(message);
        json reply = channel_.receive();
        if (reply.contains("error")) {
            throw_error_reply(reply);
        }
        return reply.at("probe");
    } catch (const ipc::IpcError& e) {
        const WorkerExit exit = reap_worker(!e.peer_closed());
        return json{{"worker_died", exit.description},
                    {"signal", exit.signal},
                    {"exit_code", exit.exit_code}};
    }
}

pid_t SandboxedEngine::worker_pid() const {
    return pid_;
}

int SandboxedEngine::landlock_abi() const {
    const std::lock_guard<std::mutex> lock(info_mutex_);
    return landlock_abi_;
}

}  // namespace lisa
