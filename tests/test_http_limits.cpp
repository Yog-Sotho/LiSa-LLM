// HTTP admission control, draining, readiness, metrics and request ids, against a
// scripted engine (tests/fake_engine.hpp): no model needed.

#include "fake_engine.hpp"
#include "server.hpp"

#include <httplib.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {

using Catch::Matchers::ContainsSubstring;

constexpr const char* kKey = "first-key-0123456789";
constexpr const char* kSecondKey = "second-key-0123456789";

lisa::Config base_config() {
    lisa::Config cfg;
    cfg.model_path = "fake.gguf";
    cfg.model_alias = "fake";
    cfg.listen_addr = "127.0.0.1";
    cfg.listen_port = 0;
    cfg.api_key = kKey;
    cfg.max_new_tokens_default = 4;
    cfg.parallel_slots = 1;  // the fake engine runs one request at a time
    return cfg;
}

class Fixture {
public:
    explicit Fixture(lisa::Config cfg = base_config()) : server_(engine_, std::move(cfg)) {
        server_.bind();
        thread_ = std::thread([this] { server_.run(); });
    }
    ~Fixture() {
        engine_.release();
        server_.stop();
        thread_.join();
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
    Fixture(Fixture&&) = delete;
    Fixture& operator=(Fixture&&) = delete;

    [[nodiscard]] httplib::Client client(const char* key = kKey) const {
        httplib::Client c("127.0.0.1", server_.port());
        c.set_read_timeout(30, 0);
        if (key != nullptr) {
            c.set_bearer_token_auth(key);
        }
        return c;
    }

    httplib::Result complete(const lisa::json& body = {{"prompt", "hi"}}) const {
        auto c = client();
        return c.Post("/v1/completions", body.dump(), "application/json");
    }

    // Starts a completion in the background that holds the model until release().
    std::future<httplib::Result> start_held_completion() {
        engine_.hold();
        auto future = std::async(std::launch::async, [this] { return complete(); });
        engine_.wait_for_generations(++generations_);
        return future;
    }

    lisa::test::FakeEngine& engine() { return engine_; }
    lisa::HttpServer& server() { return server_; }

private:
    lisa::test::FakeEngine engine_;
    lisa::HttpServer server_;
    std::thread thread_;
    int generations_ = 0;
};

lisa::json error_of(const httplib::Result& res) {
    REQUIRE(res);
    return lisa::json::parse(res->body)["error"];
}

}  // namespace

TEST_CASE("limits: a full queue answers 429 with Retry-After", "[limits]") {
    lisa::Config cfg = base_config();
    cfg.max_queued_requests = 0;  // only the running request
    Fixture fx(cfg);

    auto running = fx.start_held_completion();
    const auto refused = fx.complete();
    REQUIRE(refused);
    CHECK(refused->status == 429);
    CHECK(error_of(refused)["code"] == "server_overloaded");
    CHECK(std::stoi(refused->get_header_value("Retry-After")) >= 1);

    fx.engine().release();
    REQUIRE(running.get()->status == 200);
    CHECK(fx.complete()->status == 200);  // the model is free again
}

TEST_CASE("limits: queued requests are served in turn", "[limits]") {
    lisa::Config cfg = base_config();
    cfg.max_queued_requests = 2;
    Fixture fx(cfg);

    auto running = fx.start_held_completion();
    auto first = std::async(std::launch::async, [&] { return fx.complete(); });
    auto second = std::async(std::launch::async, [&] { return fx.complete(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // both are waiting
    const auto third = fx.complete();                             // line is full
    REQUIRE(third);
    CHECK(third->status == 429);

    fx.engine().release();
    CHECK(running.get()->status == 200);
    CHECK(first.get()->status == 200);
    CHECK(second.get()->status == 200);
}

TEST_CASE("limits: per-key rate limit", "[limits]") {
    lisa::Config cfg = base_config();
    cfg.api_keys = {kSecondKey};
    cfg.rate_limit_per_minute = 6;  // one every 10 s
    cfg.rate_limit_burst = 2;
    const Fixture fx(cfg);

    CHECK(fx.complete()->status == 200);
    CHECK(fx.complete()->status == 200);
    const auto limited = fx.complete();
    REQUIRE(limited);
    CHECK(limited->status == 429);
    CHECK(error_of(limited)["code"] == "rate_limit_exceeded");
    const int retry = std::stoi(limited->get_header_value("Retry-After"));
    CHECK(retry >= 1);
    CHECK(retry <= 10);

    // Another key has its own budget.
    auto other = fx.client(kSecondKey);
    CHECK(other.Post("/v1/completions", R"({"prompt":"hi"})", "application/json")->status == 200);
}

TEST_CASE("limits: max_prompt_tokens", "[limits]") {
    lisa::Config cfg = base_config();
    cfg.max_prompt_tokens = 8;
    const Fixture fx(cfg);
    CHECK(fx.complete({{"prompt", "short"}})->status == 200);
    const auto refused = fx.complete({{"prompt", "a prompt longer than eight tokens"}});
    REQUIRE(refused);
    CHECK(refused->status == 400);
    CHECK(error_of(refused)["code"] == "context_length_exceeded");
    CHECK(error_of(refused)["param"] == "prompt");
}

TEST_CASE("shutdown: drains running requests and refuses new ones", "[limits]") {
    Fixture fx;
    auto running = fx.start_held_completion();

    auto draining = std::async(std::launch::async, [&] { fx.server().shutdown(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto probe = fx.client(nullptr);
    const auto ready = probe.Get("/readyz");
    REQUIRE(ready);
    CHECK(ready->status == 503);
    CHECK(lisa::json::parse(ready->body)["status"] == "draining");
    const auto refused = fx.complete();
    REQUIRE(refused);
    CHECK(refused->status == 503);
    CHECK(refused->has_header("Retry-After"));

    fx.engine().release();
    const auto finished = running.get();
    REQUIRE(finished);
    CHECK(finished->status == 200);  // the running request completed normally
    draining.get();
}

TEST_CASE("shutdown: cancels what is still running at the deadline", "[limits]") {
    lisa::Config cfg = base_config();
    cfg.shutdown_timeout_seconds = 1;
    Fixture fx(cfg);
    auto running = fx.start_held_completion();  // never released

    const auto start = std::chrono::steady_clock::now();
    fx.server().shutdown();
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
    const auto cancelled = running.get();
    REQUIRE(cancelled);
    CHECK(cancelled->status == 200);
    CHECK(lisa::json::parse(cancelled->body)["choices"][0]["finish_reason"].is_null());
}

TEST_CASE("readyz reflects the engine", "[limits]") {
    Fixture fx;
    auto probe = fx.client(nullptr);
    CHECK(probe.Get("/readyz")->status == 200);
    fx.engine().set_ready(false);
    const auto unavailable = probe.Get("/readyz");
    REQUIRE(unavailable);
    CHECK(unavailable->status == 503);
    CHECK(lisa::json::parse(unavailable->body)["status"] == "unavailable");
}

TEST_CASE("metrics: Prometheus text behind the API key", "[limits]") {
    const Fixture fx;
    CHECK(fx.complete({{"prompt", "hello"}, {"max_tokens", 3}})->status == 200);
    auto anonymous = fx.client(nullptr);
    CHECK(anonymous.Get("/v1/models")->status == 401);

    const auto denied = anonymous.Get("/metrics");
    REQUIRE(denied);
    CHECK(denied->status == 401);

    auto client = fx.client();
    const auto metrics = client.Get("/metrics");
    REQUIRE(metrics);
    CHECK(metrics->status == 200);
    CHECK_THAT(metrics->get_header_value("Content-Type"), ContainsSubstring("text/plain"));
    const std::string& text = metrics->body;
    CHECK_THAT(text, ContainsSubstring("lisa_ready 1"));
    CHECK_THAT(text, ContainsSubstring(
                         R"(lisa_http_requests_total{route="/v1/completions",code="200"} 1)"));
    CHECK_THAT(text, ContainsSubstring(R"(lisa_requests_rejected_total{reason="unauthorized"} 2)"));
    CHECK_THAT(text, ContainsSubstring("lisa_completion_tokens_total 3"));
    CHECK_THAT(text, ContainsSubstring("lisa_prompt_tokens_total 5"));
    CHECK_THAT(text, ContainsSubstring(R"(lisa_queue_wait_seconds_bucket{le="+Inf"} 1)"));
    CHECK_THAT(text, ContainsSubstring("# TYPE lisa_http_request_duration_seconds histogram"));
}

TEST_CASE("request ids and security headers", "[limits]") {
    lisa::Config cfg = base_config();
    cfg.api_keys = {"sha256:" + std::string(64, '0')};  // a digest entry next to api_key
    const Fixture fx(cfg);
    auto client = fx.client();

    const auto generated = client.Get("/v1/models");
    REQUIRE(generated);
    CHECK(generated->status == 200);
    const std::string id = generated->get_header_value("X-Request-Id");
    CHECK(id.size() == 32);
    CHECK(generated->get_header_value("X-Content-Type-Options") == "nosniff");
    CHECK(generated->get_header_value("Cache-Control") == "no-store");
    CHECK_FALSE(generated->has_header("Strict-Transport-Security"));  // plain HTTP

    const auto echoed =
        client.Get("/v1/models", httplib::Headers{{"X-Request-Id", "trace-42.a_b:c"}});
    CHECK(echoed->get_header_value("X-Request-Id") == "trace-42.a_b:c");

    // Ids that could break log parsing are replaced.
    const auto replaced = client.Get("/v1/models", httplib::Headers{{"X-Request-Id", "bad id\"}"}});
    CHECK(replaced->get_header_value("X-Request-Id").size() == 32);

    // Errors carry the id too.
    auto anonymous = fx.client(nullptr);
    CHECK(anonymous.Get("/v1/models")->has_header("X-Request-Id"));
}

TEST_CASE("metrics: a streamed request is counted when its stream ends", "[limits]") {
    const Fixture fx;
    const auto streamed = fx.complete({{"prompt", "hello"}, {"max_tokens", 2}, {"stream", true}});
    REQUIRE(streamed);
    CHECK(streamed->status == 200);
    CHECK_THAT(streamed->body, ContainsSubstring("[DONE]"));

    auto client = fx.client();
    const std::string text = client.Get("/metrics")->body;
    CHECK_THAT(text, ContainsSubstring(
                         R"(lisa_http_requests_total{route="/v1/completions",code="200"} 1)"));
    CHECK_THAT(text, ContainsSubstring("lisa_completion_tokens_total 2"));
}

TEST_CASE("health_check reports readiness", "[limits]") {
    Fixture fx;
    lisa::Config cfg = base_config();
    cfg.listen_addr = "0.0.0.0";  // a wildcard address is checked through loopback
    cfg.listen_port = fx.server().port();
    std::ostringstream out;
    CHECK(lisa::health_check(cfg, out) == 0);
    CHECK_THAT(out.str(), ContainsSubstring("healthy"));

    fx.engine().set_ready(false);
    std::ostringstream not_ready;
    CHECK(lisa::health_check(cfg, not_ready) == 1);
    CHECK_THAT(not_ready.str(), ContainsSubstring("unhealthy"));

    cfg.listen_port = 1;  // nothing listens there
    std::ostringstream down;
    CHECK(lisa::health_check(cfg, down) == 1);
}
