#include "server.hpp"
#include "cpp-httplib.h"
#include <chrono>
#include <thread>

using namespace httplib;

HttpServer::HttpServer(InferenceEngine& engine, const Config& cfg)
    : engine_(engine), cfg_(cfg) {}

void HttpServer::start() {
    Server svr;
    
    svr.Get("/healthz", [](const Request&, Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    
    svr.Post("/v1/completions", [this](const Request& req, Response& res) {
        if (!cfg_.api_key.empty()) {
            auto it = req.headers.find("X-API-Key");
            if (it == req.headers.end() || it->second != cfg_.api_key) {
                res.status = 401;
                res.set_content("{\"error\":\"unauthorized\"}", "application/json");
                return;
            }
        }
        json payload;
        try {
            payload = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        std::string prompt = payload.value("prompt", "");
        int max_new = payload.value("max_new_tokens", cfg_.max_new_tokens_default);
        float temperature = payload.value("temperature", cfg_.temperature_default);
        float top_p = payload.value("top_p", cfg_.top_p_default);
        if (prompt.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt required\"}", "application/json");
            return;
        }
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        try {
            auto tokens = engine_.generate(prompt, max_new, temperature, top_p);
            for (const auto& token : tokens) {
                json data = {{"token", token}};
                std::string event = "data: " + data.dump() + "\n\n";
                if (!res.send(event.data(), event.size())) break;
            }
            res.send("data: [DONE]\n\n");
        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.send("data: " + err.dump() + "\n\n");
        }
    });
    
    std::string bind = cfg_.listen_addr + ":" + std::to_string(cfg_.listen_port);
    log_info("Starting server", {{"bind", bind}, {"tls", cfg_.enable_tls}});
    if (cfg_.enable_tls) {
        if (!svr.set_ssl_cert_file_and_private_key_file(cfg_.cert_file, cfg_.key_file))
            throw std::runtime_error("SSL init failed");
        svr.listen(cfg_.listen_addr.c_str(), cfg_.listen_port);
    } else {
        svr.listen(cfg_.listen_addr.c_str(), cfg_.listen_port);
    }
}
