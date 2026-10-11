#pragma once

#include "inference.hpp"
#include "request.hpp"
#include "utils.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace httplib {
class Server;
struct Request;
struct Response;
}  // namespace httplib

namespace lisa {

// OpenAI-compatible HTTP API: /v1/completions, /v1/chat/completions, /v1/models and
// /healthz.
class HttpServer {
public:
    HttpServer(InferenceEngine& engine, Config cfg);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;
    HttpServer(HttpServer&&) = delete;
    HttpServer& operator=(HttpServer&&) = delete;

    // Binds the listening socket (listen_port 0 picks a free port). Throws
    // std::runtime_error when the address cannot be bound.
    void bind();

    // Port the server is bound to, or -1 before bind().
    [[nodiscard]] int port() const noexcept { return port_; }

    // Binds if needed and serves until stop() is called.
    void run();

    // Thread-safe; makes run() return.
    void stop() noexcept;

private:
    enum class Api : std::uint8_t { completion, chat };

    void register_routes();
    [[nodiscard]] bool authorized(const httplib::Request& req) const;
    void respond(httplib::Response& res, Api api, std::vector<Token> prompt,
                 GenerationParams params);

    InferenceEngine& engine_;
    Config cfg_;
    std::string model_id_;
    std::int64_t created_ = 0;
    std::unique_ptr<httplib::Server> server_;
    int port_ = -1;
    std::atomic<bool> stop_requested_{false};
};

}  // namespace lisa
