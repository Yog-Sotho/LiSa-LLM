#pragma once

#include "utils.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

// Broker <-> worker messaging: JSON objects in length-prefixed frames over a Unix stream
// socket (4-byte big-endian length, then the UTF-8 JSON payload).
namespace lisa::ipc {

// Larger than any request the HTTP layer accepts (1 MiB body) once tokenised.
inline constexpr std::size_t kMaxFrameBytes = 16U << 20;

class IpcError : public std::runtime_error {
public:
    IpcError(const std::string& message, bool peer_closed)
        : std::runtime_error(message), peer_closed_(peer_closed) {}

    // True when the other side closed the connection (e.g. the worker died).
    [[nodiscard]] bool peer_closed() const noexcept { return peer_closed_; }

private:
    bool peer_closed_;
};

// Owns one end of a connected socket.
class Channel {
public:
    Channel() = default;
    explicit Channel(int fd) noexcept : fd_(fd) {}
    ~Channel();

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    Channel(Channel&& other) noexcept;
    Channel& operator=(Channel&& other) noexcept;

    // Throws IpcError. Never raises SIGPIPE.
    void send(const json& message);

    // Blocks until a whole frame arrives. Throws IpcError on EOF, oversized frames or
    // invalid JSON.
    [[nodiscard]] json receive();

    // True when a frame (or EOF) can be read within `timeout_ms` (0 = just check).
    [[nodiscard]] bool readable(int timeout_ms) const;

    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }
    [[nodiscard]] int fd() const noexcept { return fd_; }
    void close() noexcept;
    // Wakes a thread blocked in receive() (it sees EOF) without closing the descriptor
    // under it. Safe to call from another thread.
    void shutdown() const noexcept;

private:
    int fd_ = -1;
};

}  // namespace lisa::ipc
