#include "request.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <stdexcept>

namespace {

lisa::Config test_config() {
    lisa::Config cfg;
    cfg.model_path = "m.gguf";
    cfg.max_context = 1024;
    cfg.max_new_tokens_default = 64;
    cfg.temperature_default = 0.7f;
    cfg.top_p_default = 0.9f;
    return cfg;
}

}  // namespace

TEST_CASE("parse_completion_request fills defaults", "[request]") {
    const auto req = lisa::parse_completion_request(R"({"prompt":"hi"})", test_config());
    CHECK(req.prompt == "hi");
    CHECK(req.max_new_tokens == 64);
    CHECK(req.temperature == 0.7f);
    CHECK(req.top_p == 0.9f);
}

TEST_CASE("parse_completion_request reads explicit values", "[request]") {
    const auto req = lisa::parse_completion_request(
        R"({"prompt":"hi","max_new_tokens":5,"temperature":0,"top_p":1})", test_config());
    CHECK(req.max_new_tokens == 5);
    CHECK(req.temperature == 0.0f);
    CHECK(req.top_p == 1.0f);
}

TEST_CASE("parse_completion_request rejects bad input", "[request]") {
    using Catch::Matchers::ContainsSubstring;
    const lisa::Config cfg = test_config();
    const auto rejects = [&](const char* body, const char* fragment) {
        CHECK_THROWS_WITH(lisa::parse_completion_request(body, cfg), ContainsSubstring(fragment));
    };

    rejects("{not json", "invalid JSON");
    rejects("[1,2]", "JSON object");
    rejects("{}", "prompt");
    rejects(R"({"prompt":""})", "prompt");
    rejects(R"({"prompt":42})", "prompt");
    rejects(R"({"prompt":"x","max_new_tokens":0})", "max_new_tokens");
    rejects(R"({"prompt":"x","max_new_tokens":-3})", "max_new_tokens");
    rejects(R"({"prompt":"x","max_new_tokens":1000000})", "max_new_tokens");
    rejects(R"({"prompt":"x","max_new_tokens":2.5})", "max_new_tokens");
    rejects(R"({"prompt":"x","max_new_tokens":"10"})", "max_new_tokens");
    rejects(R"({"prompt":"x","temperature":-0.1})", "temperature");
    rejects(R"({"prompt":"x","temperature":3})", "temperature");
    rejects(R"({"prompt":"x","temperature":"hot"})", "temperature");
    rejects(R"({"prompt":"x","top_p":0})", "top_p");
    rejects(R"({"prompt":"x","top_p":1.5})", "top_p");
}
