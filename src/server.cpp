#include "server.hpp"

#include "openai.hpp"
#include "request.hpp"

#include <httplib.h>
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/ssl.h>
#endif
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace lisa {

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr std::size_t kMaxRequestBodyBytes = 1U << 20;  // 1 MiB
constexpr std::string_view kSseDone = "data: [DONE]\n\n";
constexpr std::string_view kBearerPrefix = "Bearer ";
constexpr const char* kJson = "application/json";
constexpr const char* kPrometheusText = "text/plain; version=0.0.4; charset=utf-8";
constexpr std::size_t kRequestIdHexDigits = 32;
constexpr std::size_t kMaxClientRequestIdLength = 64;
constexpr std::size_t kMaxLoggedPathLength = 256;
constexpr std::size_t kBaseThreads = 8;
constexpr std::size_t kSpareThreads = 8;  // health checks and refusals while the queue is full
constexpr std::size_t kMaxPendingConnections = 256;
constexpr auto kCancelGrace = std::chrono::seconds(5);
constexpr int kDrainingRetryAfterSeconds = 5;
constexpr double kMicrosPerSecond = 1'000'000.0;
// Mozilla "intermediate" TLS 1.2 suites: forward secrecy and AEAD only. TLS 1.3 suites
// are all acceptable and keep OpenSSL's defaults.
constexpr const char* kTls12Ciphers =
    "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
    "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:"
    "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305";

constexpr std::array<std::string_view, 6> kRoutes = {
    "/healthz", "/readyz", "/metrics", "/v1/models", "/v1/completions", "/v1/chat/completions"};

// Per-request data shared by the routing hook, the handler and the access logger.
// httplib runs all three for a request on the same thread, one request at a time.
struct RequestContext {
    bool active = false;
    std::string id;
    std::string client = "-";  // API key id ("key-…"), or the address without auth
    SteadyClock::time_point start;
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    bool streaming = false;  // logged when the stream ends, not by httplib's logger
};
thread_local RequestContext t_request;  // NOLINT(misc-use-anonymous-namespace)

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

double seconds_between(SteadyClock::time_point from, SteadyClock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

bool is_probe(std::string_view route) {
    return route == "/healthz" || route == "/readyz" || route == "/metrics";
}

// A client-supplied X-Request-Id is kept when it is short and plain; otherwise (or when
// absent) a random one is used, so log lines can't be forged through it.
std::string request_id_for(const httplib::Request& req) {
    const std::string given = req.get_header_value("X-Request-Id");
    const bool usable =
        !given.empty() && given.size() <= kMaxClientRequestIdLength &&
        std::all_of(given.begin(), given.end(), [](unsigned char c) {
            return std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.' || c == ':';
        });
    return usable ? given : openai::random_hex(kRequestIdHexDigits);
}

std::string presented_key(const httplib::Request& req) {
    std::string key = req.get_header_value("X-API-Key");
    if (key.empty()) {
        const std::string auth = req.get_header_value("Authorization");
        if (auth.starts_with(kBearerPrefix)) {
            key = auth.substr(kBearerPrefix.size());
        }
    }
    return key;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void harden_tls(httplib::SSLServer& server, const Config& cfg) {
    auto* ctx = static_cast<SSL_CTX*>(server.tls_context());
    // httplib already refuses anything older than TLS 1.2.
    if (cfg.tls_min_version == "1.3" && SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) != 1) {
        throw std::runtime_error("cannot require TLS 1.3");
    }
    if (SSL_CTX_set_cipher_list(ctx, kTls12Ciphers) != 1) {
        throw std::runtime_error("cannot set the TLS 1.2 cipher suites");
    }
    SSL_CTX_set_options(ctx, SSL_OP_NO_RENEGOTIATION);
}
#endif

std::unique_ptr<httplib::Server> make_server(const Config& cfg) {
    if (!cfg.enable_tls) {
        return std::make_unique<httplib::Server>();
    }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    // A client CA turns on mutual TLS: clients without a certificate it signed are refused.
    auto server = std::make_unique<httplib::SSLServer>(
        cfg.cert_file.c_str(), cfg.key_file.c_str(),
        cfg.client_ca_file.empty() ? nullptr : cfg.client_ca_file.c_str());
    if (!server->is_valid()) {
        throw std::runtime_error(
            "TLS initialisation failed: check cert_file, key_file and client_ca_file");
    }
    harden_tls(*server, cfg);
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

std::string_view route_of(std::string_view path) {
    const auto it = std::find(kRoutes.begin(), kRoutes.end(), path);
    return it == kRoutes.end() ? std::string_view("other") : *it;
}

// Timing of one generation, for metrics.
class GenerationTimer {
public:
    void piece() {
        if (!first_piece_) {
            first_piece_ = SteadyClock::now();
        }
    }
    [[nodiscard]] double first_token_seconds() const {
        return first_piece_ ? seconds_between(start_, *first_piece_) : 0.0;
    }
    [[nodiscard]] double elapsed_seconds() const {
        return seconds_between(start_, SteadyClock::now());
    }

private:
    SteadyClock::time_point start_ = SteadyClock::now();
    std::optional<SteadyClock::time_point> first_piece_;
};

}  // namespace

HttpServer::HttpServer(Engine& engine, Config cfg)
    : engine_(engine),
      cfg_(std::move(cfg)),
      keys_(KeyStore::from_config(cfg_)),
      model_id_(default_model_id(cfg_)),
      created_(std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()),
      queue_(static_cast<std::size_t>(cfg_.max_queued_requests)),
      rate_limiter_(cfg_.rate_limit_per_minute, cfg_.rate_limit_burst),
      server_(make_server(cfg_)) {
    configure_server();
    register_routes();
}

HttpServer::~HttpServer() = default;

void HttpServer::configure_server() {
    server_->set_payload_max_length(kMaxRequestBodyBytes);
    server_->set_read_timeout(std::chrono::seconds(cfg_.read_timeout_seconds));
    server_->set_write_timeout(std::chrono::seconds(cfg_.write_timeout_seconds));
    server_->set_keep_alive_timeout(std::chrono::seconds(cfg_.keep_alive_timeout_seconds));

    // Every queued request holds a thread while it waits, so the pool grows past the queue
    // size: health checks, /metrics and refusals must still get a thread.
    const std::size_t max_threads = std::max(
        kBaseThreads, static_cast<std::size_t>(cfg_.max_queued_requests) + 1 + kSpareThreads);
    server_->new_task_queue = [max_threads] {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): httplib takes ownership
        return new httplib::ThreadPool(kBaseThreads, max_threads, kMaxPendingConnections);
    };
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
                log_error("request failed", {{"error", e.what()}, {"request_id", t_request.id}});
            } catch (...) {
                log_error("request failed", {{"request_id", t_request.id}});
            }
            // Internal details stay in the log.
            send_error(res, 500, "internal server error", "server_error");
        });
    server_->set_logger([this](const httplib::Request& req, const httplib::Response& res) {
        log_request(req, res);
    });

    server_->set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        t_request = RequestContext{};
        t_request.active = true;
        t_request.id = request_id_for(req);
        t_request.start = SteadyClock::now();
        t_request.client = req.remote_addr;
        res.set_header("X-Request-Id", t_request.id);
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_header("Cache-Control", "no-store");
        if (cfg_.enable_tls) {
            res.set_header("Strict-Transport-Security", "max-age=31536000");
        }

        // The API and the metrics require a key (when keys are configured).
        if (keys_.empty() || !(req.path.starts_with("/v1/") || req.path == "/metrics")) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (std::optional<std::string> id = keys_.authenticate(presented_key(req))) {
            t_request.client = std::move(*id);
            return httplib::Server::HandlerResponse::Unhandled;
        }
        metrics_.rejected("unauthorized");
        res.set_header("WWW-Authenticate", "Bearer");
        send_error(res, 401, "invalid API key", "invalid_request_error", {}, "invalid_api_key");
        return httplib::Server::HandlerResponse::Handled;
    });

    server_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", kJson);
    });

    server_->Get("/readyz", [this](const httplib::Request&, httplib::Response& res) {
        if (draining_) {
            res.status = 503;
            res.set_content(R"({"status":"draining"})", kJson);
        } else if (!engine_.status().ready) {
            res.status = 503;
            res.set_content(
                R"({"status":"unavailable","reason":"the inference worker keeps crashing"})",
                kJson);
        } else {
            res.set_content(R"({"status":"ready"})", kJson);
        }
    });

    server_->Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        MetricsGauges gauges;
        gauges.queued = queue_.waiting();
        gauges.busy = queue_.busy();
        gauges.draining = draining_;
        gauges.engine = engine_.status();
        res.set_content(metrics_.render(gauges), kPrometheusText);
    });

    server_->Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(dump(openai::models(model_id_, created_)), kJson);
    });

    server_->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handle_generation(req, res, Api::completion);
    });
    server_->Post("/v1/chat/completions",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_generation(req, res, Api::chat);
                  });
}

void HttpServer::handle_generation(const httplib::Request& req, httplib::Response& res, Api api) {
    if (draining_) {
        metrics_.rejected("draining");
        res.set_header("Retry-After", std::to_string(kDrainingRetryAfterSeconds));
        send_error(res, 503, "the server is shutting down", "server_error", {},
                   "service_unavailable");
        return;
    }
    if (const int wait = rate_limiter_.acquire(t_request.client); wait > 0) {
        metrics_.rejected("rate_limited");
        res.set_header("Retry-After", std::to_string(wait));
        send_error(res, 429, "rate limit exceeded; retry in " + std::to_string(wait) + " s",
                   "requests", {}, "rate_limit_exceeded");
        return;
    }

    GenerationParams params;
    CompletionRequest completion;
    ChatRequest chat;
    try {
        if (api == Api::completion) {
            completion = parse_completion_request(req.body, cfg_);
            params = std::move(completion.params);
        } else {
            chat = parse_chat_request(req.body, cfg_);
            params = std::move(chat.params);
        }
    } catch (const RequestError& e) {
        send_request_error(res, e);
        return;
    }

    std::optional<AdmissionQueue::Slot> joined = queue_.try_join();
    if (!joined) {
        metrics_.rejected("queue_full");
        res.set_header("Retry-After", std::to_string(queue_.retry_after_seconds()));
        send_error(res, 429, "too many requests are waiting for the model; retry later",
                   "server_error", {}, "server_overloaded");
        return;
    }
    auto slot = std::make_shared<AdmissionQueue::Slot>(std::move(*joined));
    if (!slot->wait([&req] { return req.is_connection_closed(); })) {
        // The client left, or the server is shutting down.
        res.set_header("Retry-After", std::to_string(kDrainingRetryAfterSeconds));
        send_error(res, 503, "the request was not served", "server_error", {},
                   "service_unavailable");
        return;
    }

    std::vector<Token> prompt;
    try {
        prompt = api == Api::completion ? engine_.tokenize(completion.prompt)
                                        : engine_.tokenize_chat(chat.messages);
    } catch (const RequestError& e) {
        send_request_error(res, e);
        return;
    }
    if (cfg_.max_prompt_tokens > 0 &&
        prompt.size() > static_cast<std::size_t>(cfg_.max_prompt_tokens)) {
        send_error(res, 400,
                   "prompt (" + std::to_string(prompt.size()) + " tokens) exceeds the limit of " +
                       std::to_string(cfg_.max_prompt_tokens) + " tokens",
                   "invalid_request_error", api == Api::chat ? "messages" : "prompt",
                   "context_length_exceeded");
        return;
    }
    respond(req, res, api, std::move(prompt), std::move(params), std::move(slot));
}

void HttpServer::respond(const httplib::Request& req, httplib::Response& res, Api api,
                         std::vector<Token> prompt, GenerationParams params,
                         std::shared_ptr<AdmissionQueue::Slot> slot) {
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
    const double queue_seconds = std::chrono::duration<double>(slot->waited()).count();
    const auto record = [this, queue_seconds](const GenerationResult& result,
                                              const GenerationTimer& timer) {
        t_request.prompt_tokens = result.prompt_tokens;
        t_request.completion_tokens = result.completion_tokens;
        metrics_.generation_finished(result.prompt_tokens, result.completion_tokens, queue_seconds,
                                     timer.first_token_seconds(), timer.elapsed_seconds());
    };

    if (!params.stream) {
        std::string text;
        GenerationResult result;
        GenerationTimer timer;
        try {
            result = engine_.generate(prompt, params, [&](std::string_view piece) {
                timer.piece();
                text += piece;
                return !cancel_generations_;
            });
        } catch (const RequestError& e) {
            send_request_error(res, e);
            return;
        }
        record(result, timer);
        res.set_content(dump(chat ? openai::chat_completion(meta, text, result)
                                  : openai::completion(meta, text, result)),
                        kJson);
        return;
    }

    // httplib logs a response before writing a streamed body, so the stream logs itself
    // when the response is discarded: after the last event, or when the client leaves.
    class StreamLog {
    public:
        StreamLog(HttpServer* server, AccessRecord record)
            : server_(server), record_(std::move(record)) {}
        StreamLog(const StreamLog&) = delete;
        StreamLog& operator=(const StreamLog&) = delete;
        StreamLog(StreamLog&&) = delete;
        StreamLog& operator=(StreamLog&&) = delete;
        ~StreamLog() {
            try {
                server_->finish_request(record_);
            } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch): logging only
            }
        }
        void set_tokens(const GenerationResult& result) {
            record_.prompt_tokens = result.prompt_tokens;
            record_.completion_tokens = result.completion_tokens;
        }

    private:
        HttpServer* server_;
        AccessRecord record_;
    };
    t_request.streaming = true;
    AccessRecord access;
    access.request_id = t_request.id;
    access.method = req.method;
    access.path = req.path;
    access.remote_addr = req.remote_addr;
    access.client = t_request.client;
    access.start = t_request.start;
    access.status = 200;
    auto stream_log = std::make_shared<StreamLog>(this, std::move(access));
    const auto stream_record = [stream_log, record](const GenerationResult& result,
                                                    const GenerationTimer& timer) {
        stream_log->set_tokens(result);
        record(result, timer);
    };

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, chat, stream_record, request_id = t_request.id, meta = std::move(meta),
         prompt = std::move(prompt), params = std::move(params),
         slot = std::move(slot)](std::size_t /*offset*/, httplib::DataSink& sink) mutable {
            if (chat) {
                write_sse(sink, openai::chat_chunk(meta, {{"role", "assistant"}, {"content", ""}}));
            }
            GenerationResult result;
            GenerationTimer timer;
            try {
                result = engine_.generate(prompt, params, [&](std::string_view piece) {
                    timer.piece();
                    if (cancel_generations_ || !sink.is_writable()) {
                        return false;
                    }
                    const std::string text(piece);
                    return write_sse(sink, chat ? openai::chat_chunk(meta, {{"content", text}})
                                                : openai::completion_chunk(meta, text));
                });
                stream_record(result, timer);
            } catch (const std::exception& e) {
                log_error("generation failed", {{"error", e.what()}, {"request_id", request_id}});
                write_sse(sink, openai::error("generation failed", "server_error"));
                result.reason = StopReason::cancelled;
            }
            slot.reset();  // the model is free for the next request
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

void HttpServer::log_request(const httplib::Request& req, const httplib::Response& res) {
    if (!t_request.streaming) {
        AccessRecord record;
        if (t_request.active) {
            record.request_id = t_request.id;
            record.client = t_request.client;
            record.start = t_request.start;
        } else {  // refused before routing (e.g. 413)
            record.client = req.remote_addr;
            record.start = SteadyClock::now();
        }
        record.method = req.method;
        record.path = req.path;
        record.remote_addr = req.remote_addr;
        record.status = res.status;
        record.prompt_tokens = t_request.prompt_tokens;
        record.completion_tokens = t_request.completion_tokens;
        finish_request(record);
    }
    t_request = RequestContext{};
}

void HttpServer::finish_request(const AccessRecord& record) {
    const std::string_view route = route_of(record.path);
    const double seconds = seconds_between(record.start, SteadyClock::now());
    metrics_.request_finished(route, record.status, seconds);
    if (is_probe(route) && record.status < 400) {
        return;  // keep health checks and scrapes out of the log
    }
    json entry = {{"request_id", record.request_id},
                  {"method", record.method},
                  {"path", record.path.substr(0, kMaxLoggedPathLength)},
                  {"status", record.status},
                  {"duration_us", std::llround(seconds * kMicrosPerSecond)},
                  {"remote_addr", record.remote_addr},
                  {"client", record.client}};
    if (record.prompt_tokens > 0 || record.completion_tokens > 0) {
        entry["prompt_tokens"] = record.prompt_tokens;
        entry["completion_tokens"] = record.completion_tokens;
    }
    log_info("request", entry);
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
                                 {"mtls", !cfg_.client_ca_file.empty()},
                                 {"api_keys", keys_.size()},
                                 {"model", model_id_}});
    server_->listen_after_bind();
}

void HttpServer::shutdown() {
    draining_ = true;
    log_info("Draining requests", {{"running", queue_.busy()},
                                   {"queued", queue_.waiting()},
                                   {"timeout_seconds", cfg_.shutdown_timeout_seconds}});
    const auto deadline = SteadyClock::now() + std::chrono::seconds(cfg_.shutdown_timeout_seconds);
    if (!queue_.wait_idle(deadline)) {
        log_warn("Shutdown timeout reached; cancelling the remaining requests");
        cancel_generations_ = true;
        queue_.close();
        if (!queue_.wait_idle(SteadyClock::now() + kCancelGrace)) {
            log_warn("A generation did not stop in time");
        }
    }
    stop();
}

void HttpServer::stop() noexcept {
    stop_requested_ = true;
    server_->stop();
}

}  // namespace lisa
