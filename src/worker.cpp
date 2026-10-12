#include "worker.hpp"

#include "inference.hpp"
#include "protocol.hpp"
#include "sandbox.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lisa {

int serve_requests(ipc::Channel& channel, Engine& engine, ProbeHandler probe) {
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
        if (op == "cancel") {
            continue;  // arrived after the generation it was meant for had finished
        }
        try {
            if (op == "tokenize") {
                channel.send({{"tokens", engine.tokenize(request.at("text").get<std::string>())}});
            } else if (op == "tokenize_chat") {
                const auto messages = request.at("messages").get<std::vector<ChatMessage>>();
                channel.send({{"tokens", engine.tokenize_chat(messages)}});
            } else if (op == "generate") {
                const auto tokens = request.at("tokens").get<std::vector<Token>>();
                const auto params = request.at("params").get<GenerationParams>();
                const GenerationResult result =
                    engine.generate(tokens, params, [&channel](std::string_view piece) {
                        channel.send({{"piece", std::string(piece)}});
                        // Stop as soon as the broker cancels (client disconnected).
                        while (channel.readable(0)) {
                            if (channel.receive().value("op", "") == "cancel") {
                                return false;
                            }
                        }
                        return true;
                    });
                channel.send({{"done", result}});
            } else if (op == "probe" && probe != nullptr) {
                channel.send({{"probe", probe(request)}});
            } else {
                throw std::runtime_error("unknown worker op '" + op + "'");
            }
        } catch (const ipc::IpcError&) {
            throw;  // the channel is broken; nothing more to serve
        } catch (const std::exception& e) {
            channel.send(error_reply(e));
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
