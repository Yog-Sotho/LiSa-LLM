#pragma once

#include "admission.hpp"
#include "auth.hpp"
#include "engine.hpp"
#include "metrics.hpp"
#include "request.hpp"
#include "utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace httplib {
class Server;
struct Request;
struct Response;
}  // namespace httplib

namespace lisa {

// OpenAI-compatible HTTP API: /v1/completions, /v1/chat/completions and /v1/models, plus
// /healthz, /readyz and /metrics. Requests reach the model one at a time, first come
// first served, through a bounded queue; per-key rate limits and timeouts protect it.
class HttpServer {
public:
    HttpServer(Engine& engine, Config cfg);
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

    // Graceful shutdown: refuses new inference requests (503, /readyz turns 503), lets
    // running and queued ones finish for up to shutdown_timeout_seconds, cancels the
    // rest, then stops. Blocks until done; call from a thread other than run()'s.
    void shutdown();

    // Thread-safe; makes run() return immediately.
    void stop() noexcept;

private:
    enum class Api : std::uint8_t { completion, chat };

    void configure_server();
    void register_routes();
    void handle_generation(const httplib::Request& req, httplib::Response& res, Api api);
    void respond(const httplib::Request& req, httplib::Response& res, Api api,
                 std::vector<Token> prompt, GenerationParams params,
                 std::shared_ptr<AdmissionQueue::Slot> slot);

    // One finished request, for the access log and the request metrics.
    struct AccessRecord {
        std::string request_id = "-";
        std::string method;
        std::string path;
        std::string remote_addr;
        std::string client;
        std::chrono::steady_clock::time_point start;
        int status = 0;
        std::size_t prompt_tokens = 0;
        std::size_t completion_tokens = 0;
    };

    void log_request(const httplib::Request& req, const httplib::Response& res);
    void finish_request(const AccessRecord& record);

    Engine& engine_;
    Config cfg_;
    KeyStore keys_;
    std::string model_id_;
    std::int64_t created_ = 0;
    AdmissionQueue queue_;
    RateLimiter rate_limiter_;
    Metrics metrics_;
    std::unique_ptr<httplib::Server> server_;
    int port_ = -1;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> draining_{false};
    std::atomic<bool> cancel_generations_{false};
};

// `lisa --health-check`: asks the server described by `cfg` (on this machine) whether it
// is ready (GET /readyz). Returns 0 when it answers 200, 1 otherwise. For container
// health checks, so the image needs no HTTP client. Not usable with mutual TLS.
[[nodiscard]] int health_check(const Config& cfg, std::ostream& out);

}  // namespace lisa
