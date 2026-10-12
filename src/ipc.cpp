#include "ipc.hpp"

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>

namespace lisa::ipc {

namespace {

constexpr std::size_t kHeaderBytes = 4;

std::string errno_text(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

void write_all(int fd, const char* data, std::size_t size) {
    while (size > 0) {
        const ssize_t n = ::send(fd, data, size, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            const bool closed = errno == EPIPE || errno == ECONNRESET;
            throw IpcError(errno_text("ipc send"), closed);
        }
        data += n;
        size -= static_cast<std::size_t>(n);
    }
}

// Returns false on a clean EOF before the first byte.
bool read_all(int fd, char* data, std::size_t size) {
    std::size_t done = 0;
    while (done < size) {
        const ssize_t n = ::recv(fd, data + done, size - done, 0);
        if (n == 0) {
            if (done == 0) {
                return false;
            }
            throw IpcError("ipc receive: connection closed mid-frame", true);
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw IpcError(errno_text("ipc receive"), errno == ECONNRESET);
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

}  // namespace

Channel::~Channel() {
    close();
}

Channel::Channel(Channel&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {
}

Channel& Channel::operator=(Channel&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}

void Channel::close() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void Channel::shutdown() const noexcept {
    if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_RDWR);
    }
}

// NOLINTNEXTLINE(readability-make-member-function-const): writes to the owned socket
void Channel::send(const json& message) {
    const std::string payload = message.dump(-1, ' ', false, json::error_handler_t::replace);
    if (payload.size() > kMaxFrameBytes) {
        throw IpcError("ipc send: message exceeds the frame limit", false);
    }
    const auto size = static_cast<std::uint32_t>(payload.size());
    const std::array<char, kHeaderBytes> header = {
        static_cast<char>((size >> 24U) & 0xFFU), static_cast<char>((size >> 16U) & 0xFFU),
        static_cast<char>((size >> 8U) & 0xFFU), static_cast<char>(size & 0xFFU)};
    write_all(fd_, header.data(), header.size());
    write_all(fd_, payload.data(), payload.size());
}

// NOLINTNEXTLINE(readability-make-member-function-const): consumes from the owned socket
json Channel::receive() {
    std::array<unsigned char, kHeaderBytes> header{};
    if (!read_all(fd_, reinterpret_cast<char*>(header.data()), header.size())) {
        throw IpcError("ipc receive: connection closed", true);
    }
    const std::uint32_t size = (static_cast<std::uint32_t>(header[0]) << 24U) |
                               (static_cast<std::uint32_t>(header[1]) << 16U) |
                               (static_cast<std::uint32_t>(header[2]) << 8U) |
                               static_cast<std::uint32_t>(header[3]);
    if (size > kMaxFrameBytes) {
        throw IpcError("ipc receive: frame exceeds the size limit", false);
    }
    std::string payload(size, '\0');
    if (size > 0 && !read_all(fd_, payload.data(), payload.size())) {
        throw IpcError("ipc receive: connection closed mid-frame", true);
    }
    json message = json::parse(payload, nullptr, /*allow_exceptions=*/false);
    if (message.is_discarded() || !message.is_object()) {
        throw IpcError("ipc receive: malformed message", false);
    }
    return message;
}

bool Channel::readable(int timeout_ms) const {
    pollfd pfd{fd_, POLLIN, 0};
    while (true) {
        const int n = ::poll(&pfd, 1, timeout_ms);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return n > 0;
    }
}

}  // namespace lisa::ipc
