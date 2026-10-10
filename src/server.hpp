#pragma once

#include "inference.hpp"
#include "utils.hpp"

#include <atomic>
#include <memory>

namespace httplib {
class Server;
}

namespace lisa {

class HttpServer {
public:
    HttpServer(InferenceEngine& engine, Config cfg);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;
    HttpServer(HttpServer&&) = delete;
    HttpServer& operator=(HttpServer&&) = delete;

    // Binds and serves until stop() is called. Throws std::runtime_error when TLS
    // cannot be initialised or the address cannot be bound.
    void run();

    // Thread-safe; makes run() return.
    void stop() noexcept;

private:
    void register_routes();

    InferenceEngine& engine_;
    Config cfg_;
    std::unique_ptr<httplib::Server> server_;
    std::atomic<bool> stop_requested_{false};
};

}  // namespace lisa
