#include "server.hpp"

#include "request.hpp"

#include <httplib.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace lisa {

namespace {

constexpr std::size_t kMaxRequestBodyBytes = 1U << 20;  // 1 MiB
constexpr std::string_view kSseDone = "data: [DONE]\n\n";

void send_json_error(httplib::Response& res, int status, const std::string& message) {
    res.status = status;
    res.set_content(json{{"error", message}}.dump(), "application/json");
}

bool write_sse(httplib::DataSink& sink, const json& payload) {
    const std::string event =
        "data: " + payload.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    return sink.write(event.data(), event.size());
}

std::unique_ptr<httplib::Server> make_server(const Config& cfg) {
    if (!cfg.enable_tls) {
        return std::make_unique<httplib::Server>();
    }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    auto server = std::make_unique<httplib::SSLServer>(cfg.cert_file.c_str(), cfg.key_file.c_str());
    if (!server->is_valid()) {
        throw std::runtime_error("TLS initialisation failed: check cert_file and key_file");
    }
    return server;
#else
    throw std::runtime_error("enable_tls is set but this binary was built with LISA_TLS=OFF");
#endif
}

}  // namespace

HttpServer::HttpServer(InferenceEngine& engine, Config cfg)
    : engine_(engine), cfg_(std::move(cfg)), server_(make_server(cfg_)) {
    server_->set_payload_max_length(kMaxRequestBodyBytes);
    register_routes();
}

HttpServer::~HttpServer() = default;

void HttpServer::register_routes() {
    server_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    server_->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        if (!cfg_.api_key.empty()) {
            if (req.get_header_value("X-API-Key") != cfg_.api_key) {
                send_json_error(res, 401, "unauthorized");
                return;
            }
        }

        CompletionRequest completion;
        std::vector<Token> prompt_tokens;
        try {
            completion = parse_completion_request(req.body, cfg_);
            prompt_tokens = engine_.tokenize(completion.prompt);
        } catch (const std::invalid_argument& e) {
            send_json_error(res, 400, e.what());
            return;
        }
        const std::size_t needed =
            prompt_tokens.size() + static_cast<std::size_t>(completion.max_new_tokens);
        if (needed > engine_.info().n_ctx) {
            send_json_error(res, 400,
                            "prompt (" + std::to_string(prompt_tokens.size()) +
                                " tokens) plus max_new_tokens exceeds the context window of " +
                                std::to_string(engine_.info().n_ctx) + " tokens");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, completion = std::move(completion), prompt_tokens = std::move(prompt_tokens)](
                std::size_t /*offset*/, httplib::DataSink& sink) {
                try {
                    engine_.generate(prompt_tokens, completion, [&sink](std::string_view piece) {
                        return sink.is_writable() &&
                               write_sse(sink, json{{"token", std::string(piece)}});
                    });
                } catch (const std::exception& e) {
                    log_error("generation failed", {{"error", e.what()}});
                    write_sse(sink, json{{"error", "generation failed"}});
                }
                sink.write(kSseDone.data(), kSseDone.size());
                sink.done();
                return true;
            });
    });
}

void HttpServer::run() {
    if (stop_requested_) {
        return;
    }
    log_info("Starting server",
             {{"bind", cfg_.listen_addr + ":" + std::to_string(cfg_.listen_port)},
              {"tls", cfg_.enable_tls}});
    if (!server_->listen(cfg_.listen_addr, cfg_.listen_port) && !stop_requested_) {
        throw std::runtime_error("cannot listen on " + cfg_.listen_addr + ":" +
                                 std::to_string(cfg_.listen_port));
    }
}

void HttpServer::stop() noexcept {
    stop_requested_ = true;
    server_->stop();
}

}  // namespace lisa
