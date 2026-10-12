// HTTP API tests: a real HttpServer on an ephemeral port, driven with httplib::Client.
// They need LISA_TEST_MODEL (see test_engine.cpp) and skip otherwise.

#include "inference.hpp"
#include "server.hpp"

#include <httplib.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kKey = "test-key";

class TestServer {
public:
    explicit TestServer(const char* model_path) {
        cfg_.model_path = model_path;
        cfg_.model_alias = "tiny";
        cfg_.listen_addr = "127.0.0.1";
        cfg_.listen_port = 0;
        cfg_.api_key = kKey;
        cfg_.n_threads = 2;
        cfg_.max_context = 128;
        cfg_.max_new_tokens_default = 500;  // larger than the context: must be clamped
        cfg_.temperature_default = 0.0f;
        cfg_.use_gpu = false;
        engine_ = std::make_unique<lisa::InferenceEngine>(cfg_);
        server_ = std::make_unique<lisa::HttpServer>(*engine_, cfg_);
        server_->bind();
        thread_ = std::thread([this] { server_->run(); });
    }
    ~TestServer() {
        server_->stop();
        thread_.join();
    }
    TestServer(const TestServer&) = delete;
    TestServer& operator=(const TestServer&) = delete;
    TestServer(TestServer&&) = delete;
    TestServer& operator=(TestServer&&) = delete;

    [[nodiscard]] httplib::Client client(bool with_key = true) const {
        httplib::Client c("127.0.0.1", server_->port());
        c.set_read_timeout(60, 0);
        if (with_key) {
            c.set_bearer_token_auth(kKey);
        }
        return c;
    }

private:
    lisa::Config cfg_;
    std::unique_ptr<lisa::InferenceEngine> engine_;
    std::unique_ptr<lisa::HttpServer> server_;
    std::thread thread_;
};

const TestServer* shared_server() {
    static const std::unique_ptr<TestServer> server = [] {
        std::unique_ptr<TestServer> s;
        const char* path = std::getenv("LISA_TEST_MODEL");
        if (path != nullptr && *path != '\0') {
            s = std::make_unique<TestServer>(path);
        }
        return s;
    }();
    return server.get();
}

lisa::json post_json(httplib::Client& client, const std::string& path, const lisa::json& body,
                     int expected_status) {
    const auto res = client.Post(path, body.dump(), "application/json");
    REQUIRE(res);
    INFO(res->body);
    CHECK(res->status == expected_status);
    return lisa::json::parse(res->body);
}

// Splits an SSE body into the JSON payloads of its "data:" lines (without [DONE]).
std::vector<lisa::json> sse_events(const std::string& body, bool& saw_done) {
    std::vector<lisa::json> events;
    std::istringstream lines(body);
    std::string line;
    saw_done = false;
    while (std::getline(lines, line)) {
        if (!line.starts_with("data: ")) {
            continue;
        }
        const std::string data = line.substr(6);
        if (data == "[DONE]") {
            saw_done = true;
        } else {
            events.push_back(lisa::json::parse(data));
        }
    }
    return events;
}

#define REQUIRE_SERVER(name)                  \
    const TestServer* name = shared_server(); \
    if ((name) == nullptr) {                  \
        SKIP("LISA_TEST_MODEL not set");      \
    }

}  // namespace

TEST_CASE("HTTP: health check needs no key, /v1 routes do", "[server]") {
    REQUIRE_SERVER(server);
    auto anonymous = server->client(false);
    const auto health = anonymous.Get("/healthz");
    REQUIRE(health);
    CHECK(health->status == 200);

    const auto denied = anonymous.Get("/v1/models");
    REQUIRE(denied);
    CHECK(denied->status == 401);
    CHECK(lisa::json::parse(denied->body)["error"]["code"] == "invalid_api_key");

    const httplib::Headers wrong = {{"Authorization", "Bearer nope"}};
    CHECK(anonymous.Get("/v1/models", wrong)->status == 401);

    const httplib::Headers x_api_key = {{"X-API-Key", kKey}};
    const auto models = anonymous.Get("/v1/models", x_api_key);
    REQUIRE(models);
    CHECK(models->status == 200);
    CHECK(lisa::json::parse(models->body)["data"][0]["id"] == "tiny");
}

TEST_CASE("HTTP: unknown routes and oversized bodies get JSON errors", "[server]") {
    REQUIRE_SERVER(server);
    auto client = server->client();
    const auto missing = client.Get("/v1/nope");
    REQUIRE(missing);
    CHECK(missing->status == 404);
    CHECK(lisa::json::parse(missing->body)["error"]["type"] == "not_found_error");

    const auto huge =
        client.Post("/v1/completions", std::string(2U << 20, 'x'), "application/json");
    REQUIRE(huge);
    CHECK(huge->status == 413);
    CHECK(lisa::json::parse(huge->body).contains("error"));
}

TEST_CASE("HTTP: text completion, non-streaming", "[server]") {
    REQUIRE_SERVER(server);
    auto client = server->client();
    const auto body = post_json(client, "/v1/completions",
                                {{"model", "whatever"},
                                 {"prompt", "Once upon a time"},
                                 {"max_tokens", 6},
                                 {"temperature", 0}},
                                200);
    CHECK(body["object"] == "text_completion");
    CHECK(body["model"] == "tiny");
    CHECK(body["id"].get<std::string>().starts_with("cmpl-"));
    CHECK_FALSE(body["choices"][0]["text"].get<std::string>().empty());
    const auto finish = body["choices"][0]["finish_reason"];
    CHECK((finish == "length" || finish == "stop"));
    CHECK(body["usage"]["completion_tokens"].get<int>() <= 6);
    CHECK(body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"].get<int>() +
                                               body["usage"]["completion_tokens"].get<int>());
}

TEST_CASE("HTTP: streamed completion matches the non-streamed text", "[server]") {
    REQUIRE_SERVER(server);
    auto client = server->client();
    const lisa::json request = {
        {"prompt", "Once upon a time"}, {"max_tokens", 12}, {"temperature", 0}};
    const auto whole = post_json(client, "/v1/completions", request, 200);

    lisa::json streaming = request;
    streaming["stream"] = true;
    streaming["stream_options"] = {{"include_usage", true}};
    const auto res = client.Post("/v1/completions", streaming.dump(), "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK_THAT(res->get_header_value("Content-Type"),
               Catch::Matchers::ContainsSubstring("text/event-stream"));

    bool done = false;
    const auto events = sse_events(res->body, done);
    CHECK(done);
    REQUIRE(events.size() >= 3);
    std::string text;
    for (std::size_t i = 0; i + 2 < events.size(); ++i) {
        CHECK(events[i]["object"] == "text_completion");
        CHECK(events[i]["choices"][0]["finish_reason"].is_null());
        text += events[i]["choices"][0]["text"].get<std::string>();
    }
    const auto& final_chunk = events[events.size() - 2];
    CHECK_FALSE(final_chunk["choices"][0]["finish_reason"].is_null());
    const auto& usage_chunk = events.back();
    CHECK(usage_chunk["choices"].empty());
    for (const char* field : {"prompt_tokens", "completion_tokens", "total_tokens"}) {
        CHECK(usage_chunk["usage"][field] == whole["usage"][field]);
    }
    // The same prompt again: all but its last token come from the prompt cache.
    CHECK(usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] ==
          whole["usage"]["prompt_tokens"].get<int>() - 1);
    CHECK(text == whole["choices"][0]["text"].get<std::string>());
}

TEST_CASE("HTTP: chat completion, streaming and non-streaming", "[server]") {
    REQUIRE_SERVER(server);
    auto client = server->client();
    const lisa::json request = {{"messages",
                                 {{{"role", "system"}, {"content", "Tell a story."}},
                                  {{"role", "user"}, {"content", "Once upon a time"}}}},
                                {"max_completion_tokens", 8},
                                {"temperature", 0}};
    const auto body = post_json(client, "/v1/chat/completions", request, 200);
    CHECK(body["object"] == "chat.completion");
    CHECK(body["id"].get<std::string>().starts_with("chatcmpl-"));
    CHECK(body["choices"][0]["message"]["role"] == "assistant");
    CHECK(body["choices"][0]["message"]["content"].is_string());

    lisa::json streaming = request;
    streaming["stream"] = true;
    const auto res = client.Post("/v1/chat/completions", streaming.dump(), "application/json");
    REQUIRE(res);
    bool done = false;
    const auto events = sse_events(res->body, done);
    CHECK(done);
    REQUIRE(events.size() >= 2);
    CHECK(events.front()["object"] == "chat.completion.chunk");
    CHECK(events.front()["choices"][0]["delta"]["role"] == "assistant");
    std::string content;
    for (const auto& event : events) {
        content += event["choices"][0]["delta"].value("content", "");
    }
    CHECK(content == body["choices"][0]["message"]["content"].get<std::string>());
    CHECK_FALSE(events.back()["choices"][0]["finish_reason"].is_null());
}

TEST_CASE("HTTP: request validation errors", "[server]") {
    REQUIRE_SERVER(server);
    auto client = server->client();
    auto err = post_json(client, "/v1/completions", {{"prompt", "x"}, {"top_p", 3}}, 400);
    CHECK(err["error"]["type"] == "invalid_request_error");
    CHECK(err["error"]["param"] == "top_p");

    err = post_json(client, "/v1/chat/completions",
                    {{"messages", {{{"role", "user"}, {"content", "</s> hi"}}}}}, 400);
    CHECK(err["error"]["param"] == "messages[0].content");

    std::string long_prompt;
    for (int i = 0; i < 20; ++i) {
        long_prompt += "Once upon a time ";
    }
    err = post_json(client, "/v1/completions", {{"prompt", long_prompt}, {"max_tokens", 100}}, 400);
    CHECK(err["error"]["code"] == "context_length_exceeded");

    // Without max_tokens the configured default (500) is clamped to the free context.
    const auto ok = post_json(client, "/v1/completions", {{"prompt", long_prompt}}, 200);
    CHECK(ok["usage"]["total_tokens"].get<int>() <= 128);
}

TEST_CASE("HTTP: response_format json_object", "[server]") {
    REQUIRE_SERVER(server);
    auto client = server->client();
    const auto body = post_json(client, "/v1/chat/completions",
                                {{"messages", {{{"role", "user"}, {"content", "Reply in JSON."}}}},
                                 {"max_tokens", 40},
                                 {"temperature", 0},
                                 {"response_format", {{"type", "json_object"}}}},
                                200);
    const std::string content = body["choices"][0]["message"]["content"];
    INFO(content);
    REQUIRE_FALSE(content.empty());
    CHECK(content.front() == '{');
    if (body["choices"][0]["finish_reason"] == "stop") {
        CHECK_NOTHROW(lisa::json::parse(content));
    }
    post_json(client, "/v1/chat/completions",
              {{"messages", {{{"role", "user"}, {"content", "x"}}}},
               {"response_format", {{"type", "json_schema"}}}},
              400);
}
