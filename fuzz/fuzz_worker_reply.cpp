// libFuzzer target: bytes sent by a compromised sandboxed worker. They go through the
// broker's frame reader (ipc::Channel) and the decoders it applies to replies. Allowed
// failures are the ones the broker turns into "kill and restart the worker": IpcError
// (bad framing or JSON) and json::exception (wrong shape); RequestError/runtime_error
// come from decoding an error reply. Anything else, or a sanitizer report, is a bug.

#include "ipc.hpp"
#include "protocol.hpp"

#include <sys/socket.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMaxInput = 64U << 10;  // stays below the socket buffer size

void decode_like_the_broker(const lisa::json& message) {
    try {
        if (message.contains("error")) {
            lisa::throw_error_reply(message);
        }
        if (message.contains("tokens")) {
            (void)message.at("tokens").get<std::vector<lisa::Token>>();
        }
        if (message.contains("piece")) {
            (void)message.at("piece").get_ref<const std::string&>();
        }
        if (message.contains("done")) {
            (void)message.at("done").get<lisa::GenerationResult>();
        }
        if (message.contains("ready")) {
            (void)message.at("ready").get<lisa::ModelInfo>();
            (void)message.value("landlock_abi", 0);
        }
        if (message.contains("fatal")) {
            (void)message.at("fatal").get<std::string>();
        }
    } catch (const lisa::json::exception&) {
    } catch (const lisa::RequestError&) {
    } catch (const std::runtime_error&) {
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > kMaxInput) {
        return 0;
    }
    int fds[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) != 0) {
        return 0;
    }
    std::size_t written = 0;
    while (written < size) {
        const ssize_t n = ::write(fds[1], data + written, size - written);
        if (n <= 0) {
            break;
        }
        written += static_cast<std::size_t>(n);
    }
    ::close(fds[1]);  // EOF after the input

    lisa::ipc::Channel channel(fds[0]);
    try {
        while (true) {
            decode_like_the_broker(channel.receive());
        }
    } catch (const lisa::ipc::IpcError&) {
    }
    return 0;
}
