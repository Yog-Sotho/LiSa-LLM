// A compromised inference worker: speaks the framing correctly but sends replies of the
// wrong shape. The broker must treat them as protocol violations (kill and restart).
//
//   init                 -> valid "ready", or {"ready": 5} if chat_template is
//                           "malformed-ready"
//   tokenize             -> {"tokens": "not a list"}
//   tokenize_chat        -> a valid reply
//   generate             -> {"piece": 5}

#include "ipc.hpp"
#include "worker.hpp"

#include <exception>

int main() {
    lisa::ipc::Channel channel(lisa::kWorkerChannelFd);
    try {
        const lisa::json init = channel.receive();
        if (init.at("settings").value("chat_template", "") == "malformed-ready") {
            channel.send({{"ready", 5}});
        } else {
            channel.send({{"ready",
                           {{"description", "hostile"},
                            {"chat_template", "chatml"},
                            {"n_layer", 1},
                            {"n_ctx_train", 64},
                            {"n_ctx", 64},
                            {"n_params", 1},
                            {"gpu_offload", false}}},
                          {"landlock_abi", 0}});
        }
        while (true) {
            const lisa::json request = channel.receive();
            const std::string op = request.value("op", "");
            if (op == "tokenize") {
                channel.send({{"tokens", "not a list"}});
            } else if (op == "tokenize_chat") {
                channel.send({{"tokens", {1, 2, 3}}});
            } else if (op == "generate") {
                channel.send({{"piece", 5}});
            }
        }
    } catch (const std::exception&) {
        return 0;  // the broker hung up
    }
}
