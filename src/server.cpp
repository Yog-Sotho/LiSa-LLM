#include "server.hpp"

#include "openai.hpp"
#include "request.hpp"

#include <httplib.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace lisa {

namespace {

constexpr std::size_t kMaxRequestBodyBytes = 1U << 20;  // 1 MiB
constexpr std::string_view kSseDone = "data: [DONE]\n\n";
constexpr std::string_view kBearerPrefix = "Bearer ";
constexpr const char* kJson = "application/json";

std::string dump(const json& body) {
    // `replace` keeps invalid UTF-8 from model output from throwing mid-response.
    return body.dump(-1, ' ', false, json::error_handler_t::replace);
}

void send_error(httplib::Response& res, int status, std::string_view message, std::string_view type,
                std::string_view param = {}, std::string_view code = {}) {
    res.status = status;
    res.set_content(dump(openai::error(message, type, param, code)), kJson);
}

void send_request_error(httplib::Response& res, const RequestError& e) {
    send_error(res, 400, e.what(), "invalid_request_error", e.param());
}

bool write_sse(httplib::DataSink& sink, const json& payload) {
    const std::string event = "data: " + dump(payload) + "\n\n";
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

std::string default_model_id(const Config& cfg) {
    if (!cfg.model_alias.empty()) {
        return cfg.model_alias;
    }
    const std::string stem = std::filesystem::path(cfg.model_path).stem().string();
    return stem.empty() ? "model" : stem;
}

}  // namespace

HttpServer::HttpServer(Engine& engine, Config cfg)
    : engine_(engine),
      cfg_(std::move(cfg)),
      model_id_(default_model_id(cfg_)),
      created_(std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()),
      server_(make_server(cfg_)) {
    server_->set_payload_max_length(kMaxRequestBodyBytes);
    register_routes();
}

HttpServer::~HttpServer() = default;

bool HttpServer::authorized(const httplib::Request& req) const {
    if (cfg_.api_key.empty()) {
        return true;
    }
    std::string key = req.get_header_value("X-API-Key");
    if (key.empty()) {
        const std::string auth = req.get_header_value("Authorization");
        if (auth.starts_with(kBearerPrefix)) {
            key = auth.substr(kBearerPrefix.size());
        }
    }
    return constant_time_equals(key, cfg_.api_key);
}

void HttpServer::register_routes() {
    // Unknown routes, 413 (body too large) etc. get an OpenAI-style JSON body too.
    server_->set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.body.empty()) {
            send_error(res, res.status, httplib::status_message(res.status),
                       res.status == 404 ? "not_found_error" : "invalid_request_error");
        }
    });
    server_->set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, const std::exception_ptr& ep) {
            try {
                std::rethrow_exception(ep);
            } catch (const std::exception& e) {
                log_error("request failed", {{"error", e.what()}});
            } catch (...) {
                log_error("request failed");
            }
            send_error(res, 500, "internal server error", "server_error");
        });

    server_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", kJson);
    });

    // Every /v1 route requires the API key.
    server_->set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (req.path.starts_with("/v1/") && !authorized(req)) {
            res.set_header("WWW-Authenticate", "Bearer");
            send_error(res, 401, "invalid API key", "invalid_request_error", {}, "invalid_api_key");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_->Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(dump(openai::models(model_id_, created_)), kJson);
    });

    server_->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        CompletionRequest request;
        std::vector<Token> prompt;
        try {
            request = parse_completion_request(req.body, cfg_);
            prompt = engine_.tokenize(request.prompt);
        } catch (const RequestError& e) {
            send_request_error(res, e);
            return;
        }
        respond(res, Api::completion, std::move(prompt), std::move(request.params));
    });

    server_->Post("/v1/chat/completions",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      ChatRequest request;
                      std::vector<Token> prompt;
                      try {
                          request = parse_chat_request(req.body, cfg_);
                          prompt = engine_.tokenize_chat(request.messages);
                      } catch (const RequestError& e) {
                          send_request_error(res, e);
                          return;
                      }
                      respond(res, Api::chat, std::move(prompt), std::move(request.params));
                  });
}

void HttpServer::respond(httplib::Response& res, Api api, std::vector<Token> prompt,
                         GenerationParams params) {
    const std::size_t n_ctx = engine_.info().n_ctx;
    if (prompt.size() >= n_ctx) {
        send_error(res, 400,
                   "prompt (" + std::to_string(prompt.size()) +
                       " tokens) does not fit the context window of " + std::to_string(n_ctx) +
                       " tokens",
                   "invalid_request_error", "messages", "context_length_exceeded");
        return;
    }
    const std::size_t room = n_ctx - prompt.size();
    if (!params.max_tokens_explicit) {
        // Like OpenAI: without max_tokens, generate at most until the context is full.
        params.max_tokens =
            static_cast<int>(std::min(static_cast<std::size_t>(params.max_tokens), room));
    } else if (static_cast<std::size_t>(params.max_tokens) > room) {
        send_error(res, 400,
                   "prompt (" + std::to_string(prompt.size()) + " tokens) plus max_tokens (" +
                       std::to_string(params.max_tokens) + ") exceeds the context window of " +
                       std::to_string(n_ctx) + " tokens",
                   "invalid_request_error", "max_tokens", "context_length_exceeded");
        return;
    }

    const bool chat = api == Api::chat;
    openai::ResponseMeta meta = openai::make_meta(chat ? "chatcmpl-" : "cmpl-", model_id_);

    if (!params.stream) {
        std::string text;
        GenerationResult result;
        try {
            result = engine_.generate(prompt, params, [&text](std::string_view piece) {
                text += piece;
                return true;
            });
        } catch (const RequestError& e) {
            send_request_error(res, e);
            return;
        }
        res.set_content(dump(chat ? openai::chat_completion(meta, text, result)
                                  : openai::completion(meta, text, result)),
                        kJson);
        return;
    }

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, chat, meta = std::move(meta), prompt = std::move(prompt),
         params = std::move(params)](std::size_t /*offset*/, httplib::DataSink& sink) {
            if (chat) {
                write_sse(sink, openai::chat_chunk(meta, {{"role", "assistant"}, {"content", ""}}));
            }
            GenerationResult result;
            try {
                result = engine_.generate(prompt, params, [&](std::string_view piece) {
                    if (!sink.is_writable()) {
                        return false;
                    }
                    const std::string text(piece);
                    return write_sse(sink, chat ? openai::chat_chunk(meta, {{"content", text}})
                                                : openai::completion_chunk(meta, text));
                });
            } catch (const std::exception& e) {
                log_error("generation failed", {{"error", e.what()}});
                write_sse(sink, openai::error("generation failed", "server_error"));
                result.reason = StopReason::cancelled;
            }
            if (result.reason != StopReason::cancelled) {
                const json finish = openai::finish_reason(result.reason);
                write_sse(sink, chat ? openai::chat_chunk(meta, json::object(), finish)
                                     : openai::completion_chunk(meta, "", finish));
                if (params.include_usage) {
                    write_sse(sink, openai::usage_chunk(
                                        meta, chat ? "chat.completion.chunk" : "text_completion",
                                        result));
                }
            }
            sink.write(kSseDone.data(), kSseDone.size());
            sink.done();
            return true;
        });
}

void HttpServer::bind() {
    if (port_ >= 0) {
        return;
    }
    const std::string where = cfg_.listen_addr + ":" + std::to_string(cfg_.listen_port);
    if (cfg_.listen_port == 0) {
        port_ = server_->bind_to_any_port(cfg_.listen_addr);
    } else if (server_->bind_to_port(cfg_.listen_addr, cfg_.listen_port)) {
        port_ = cfg_.listen_port;
    }
    if (port_ < 0) {
        throw std::runtime_error("cannot listen on " + where);
    }
}

void HttpServer::run() {
    bind();
    if (stop_requested_) {
        return;
    }
    log_info("Starting server", {{"bind", cfg_.listen_addr + ":" + std::to_string(port_)},
                                 {"tls", cfg_.enable_tls},
                                 {"model", model_id_}});
    server_->listen_after_bind();
}

void HttpServer::stop() noexcept {
    stop_requested_ = true;
    server_->stop();
}

}  // namespace lisa
