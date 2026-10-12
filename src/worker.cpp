#include "worker.hpp"

#include "inference.hpp"
#include "protocol.hpp"
#include "sandbox.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lisa {

namespace {

// State shared with the engine's scheduler thread, which sends generation output.
// Held by shared_ptr because callbacks may still run while the engine shuts down.
struct Outbox {
    explicit Outbox(ipc::Channel& c) : channel(c) {}

    // Never throws: once the broker is gone, output is dropped.
    void send(const json& message) noexcept {
        const std::lock_guard<std::mutex> lock(mutex);
        if (broken) {
            return;
        }
        try {
            channel.send(message);
        } catch (const std::exception&) {
            broken = true;
        }
    }

    ipc::Channel& channel;
    std::mutex mutex;  // one writer at a time; guards the members below
    bool broken = false;
    // Broker request id -> engine request id, for cancellation.
    std::unordered_map<std::uint64_t, InferenceEngine::RequestId> running;
};

json tagged(json message, std::uint64_t id) {
    message["id"] = id;
    return message;
}

}  // namespace

int serve_requests(ipc::Channel& channel, InferenceEngine& engine, ProbeHandler probe) {
    const auto outbox = std::make_shared<Outbox>(channel);
    while (true) {
        json request;
        try {
            request = channel.receive();
        } catch (const ipc::IpcError& e) {
            if (e.peer_closed()) {
                return 0;
            }
            throw;
        }

        const std::string op = request.value("op", "");
        const auto id = request.value("id", std::uint64_t{0});
        try {
            if (op == "cancel") {
                std::optional<InferenceEngine::RequestId> target;
                {
                    const std::lock_guard<std::mutex> lock(outbox->mutex);
                    if (const auto it = outbox->running.find(id); it != outbox->running.end()) {
                        target = it->second;
                    }
                }
                if (target) {
                    engine.cancel(*target);
                }  // else: it had already finished
            } else if (op == "tokenize") {
                outbox->send(tagged(
                    {{"tokens", engine.tokenize(request.at("text").get<std::string>())}}, id));
            } else if (op == "tokenize_chat") {
                const auto messages = request.at("messages").get<std::vector<ChatMessage>>();
                outbox->send(tagged({{"tokens", engine.tokenize_chat(messages)}}, id));
            } else if (op == "generate") {
                InferenceEngine::Callbacks callbacks;
                callbacks.on_piece = [outbox, id](std::string_view piece) {
                    outbox->send(tagged({{"piece", std::string(piece)}}, id));
                    return true;  // the broker cancels explicitly
                };
                callbacks.on_done = [outbox, id](const GenerationResult& result) {
                    {
                        const std::lock_guard<std::mutex> lock(outbox->mutex);
                        outbox->running.erase(id);
                    }
                    outbox->send(tagged({{"done", result}}, id));
                };
                callbacks.on_error = [outbox, id](const std::exception_ptr& error) {
                    {
                        const std::lock_guard<std::mutex> lock(outbox->mutex);
                        outbox->running.erase(id);
                    }
                    try {
                        std::rethrow_exception(error);
                    } catch (const std::exception& e) {
                        outbox->send(tagged(error_reply(e), id));
                    }
                };
                // Registered before the engine can finish it, so done/error find the entry.
                const std::lock_guard<std::mutex> lock(outbox->mutex);
                outbox->running[id] = engine.submit(request.at("tokens").get<std::vector<Token>>(),
                                                    request.at("params").get<GenerationParams>(),
                                                    std::move(callbacks));
            } else if (op == "probe" && probe != nullptr) {
                outbox->send(tagged({{"probe", probe(request)}}, id));
            } else {
                throw std::runtime_error("unknown worker op '" + op + "'");
            }
        } catch (const ipc::IpcError&) {
            throw;  // the channel is broken; nothing more to serve
        } catch (const std::exception& e) {
            outbox->send(tagged(error_reply(e), id));
        }
    }
}

int worker_main(ProbeHandler probe) {
    if (::fcntl(kWorkerChannelFd, F_GETFD) < 0 || ::fcntl(kWorkerModelFd, F_GETFD) < 0) {
        std::cerr << "lisa: " << kWorkerFlag << " is internal and started by the server\n";
        return 2;
    }
    // Nothing else may leak in from the broker.
    ::close_range(kWorkerModelFd + 1, ~0U, 0);

    ipc::Channel channel(kWorkerChannelFd);
    try {
        const json init = channel.receive();
        if (init.value("op", "") != "init") {
            throw std::runtime_error("expected an init message");
        }
        const auto settings = init.at("settings").get<WorkerSettings>();

        SandboxPolicy policy;
        policy.memory_limit_bytes = settings.memory_limit_bytes;
        policy.cgroup_parent = settings.cgroup;
        // From here on: own namespaces, empty read-only root with the model at
        // kSandboxModelPath, no capabilities, Landlock. Must happen before the model
        // (untrusted input) is parsed.
        enter_worker_sandbox(policy, kWorkerModelFd);
        install_seccomp(SeccompStage::load, settings.seccomp_log);

        Config cfg;
        cfg.model_path = std::string(kSandboxModelPath);
        cfg.chat_template = settings.chat_template;
        cfg.n_threads = settings.n_threads;
        cfg.max_context = settings.max_context;
        cfg.parallel_slots = settings.parallel_slots;
        cfg.flash_attention = settings.flash_attention;
        cfg.kv_cache_type = settings.kv_cache_type;
        cfg.use_gpu = false;
        InferenceEngine engine(cfg);
        engine.warm_up();
        install_seccomp(SeccompStage::serve, settings.seccomp_log);

        channel.send({{"ready", engine.info()}, {"landlock_abi", landlock_abi_in_effect()}});
        return serve_requests(channel, engine, probe);
    } catch (const std::exception& e) {
        try {
            channel.send({{"fatal", e.what()}});
        } catch (const ipc::IpcError&) {
            return 1;  // the broker is gone; nothing to report to
        }
        return 1;
    }
}

}  // namespace lisa
