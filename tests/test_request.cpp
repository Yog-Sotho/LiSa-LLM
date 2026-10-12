#include "request.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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

// Checks that parsing fails with a message containing `fragment` and the given param.
void check_rejects(const char* body, const char* fragment, const char* param, bool chat = false) {
    using Catch::Matchers::ContainsSubstring;
    const lisa::Config cfg = test_config();
    try {
        if (chat) {
            (void)lisa::parse_chat_request(body, cfg);
        } else {
            (void)lisa::parse_completion_request(body, cfg);
        }
        FAIL("expected RequestError for " << body);
    } catch (const lisa::RequestError& e) {
        CHECK_THAT(e.what(), ContainsSubstring(fragment));
        CHECK(e.param() == param);
    }
}

}  // namespace

TEST_CASE("completion request: defaults come from the config", "[request]") {
    const auto req = lisa::parse_completion_request(R"({"prompt":"hi"})", test_config());
    CHECK(req.prompt == "hi");
    CHECK(req.params.max_tokens == 64);
    CHECK_FALSE(req.params.max_tokens_explicit);
    CHECK(req.params.sampling.temperature == 0.7f);
    CHECK(req.params.sampling.top_p == 0.9f);
    CHECK(req.params.sampling.top_k == 0);
    CHECK(req.params.sampling.seed == lisa::kRandomSeed);
    CHECK(req.params.stop.empty());
    CHECK_FALSE(req.params.stream);
}

TEST_CASE("completion request: OpenAI and llama.cpp parameters", "[request]") {
    const auto req = lisa::parse_completion_request(R"({
        "model": "anything", "prompt": ["one prompt"], "max_tokens": 5, "temperature": 0,
        "top_p": 1, "top_k": 40, "min_p": 0.05, "presence_penalty": -1.5,
        "frequency_penalty": 2, "repeat_penalty": 1.1, "seed": 42, "stop": ["\n", "END"],
        "stream": true, "stream_options": {"include_usage": true}, "n": 1, "user": "u1",
        "logprobs": null, "tools": []
    })",
                                                    test_config());
    CHECK(req.prompt == "one prompt");
    CHECK(req.params.max_tokens == 5);
    CHECK(req.params.max_tokens_explicit);
    const auto& s = req.params.sampling;
    CHECK(s.temperature == 0.0f);
    CHECK(s.top_k == 40);
    CHECK(s.min_p == 0.05f);
    CHECK(s.presence_penalty == -1.5f);
    CHECK(s.frequency_penalty == 2.0f);
    CHECK(s.repeat_penalty == 1.1f);
    CHECK(s.seed == 42U);
    CHECK(req.params.stop == std::vector<std::string>{"\n", "END"});
    CHECK(req.params.stream);
    CHECK(req.params.include_usage);
}

TEST_CASE("completion request: max_tokens aliases", "[request]") {
    const auto cfg = test_config();
    CHECK(lisa::parse_completion_request(R"({"prompt":"x","max_new_tokens":7})", cfg)
              .params.max_tokens == 7);
    CHECK(lisa::parse_completion_request(R"({"prompt":"x","max_tokens":null})", cfg)
              .params.max_tokens == 64);
    CHECK(lisa::parse_chat_request(
              R"({"messages":[{"role":"user","content":"x"}],"max_completion_tokens":9})", cfg)
              .params.max_tokens == 9);
    CHECK(lisa::parse_completion_request(R"({"prompt":"x","stop":"###"})", cfg).params.stop ==
          std::vector<std::string>{"###"});
}

TEST_CASE("completion request: invalid input names the parameter", "[request]") {
    check_rejects("{not json", "invalid JSON", "");
    check_rejects("[1,2]", "JSON object", "");
    check_rejects("{}", "prompt", "prompt");
    check_rejects(R"({"prompt":""})", "prompt", "prompt");
    check_rejects(R"({"prompt":["a","b"]})", "prompt", "prompt");
    check_rejects(R"({"prompt":"x","max_tokens":0})", "between 1 and 1024", "max_tokens");
    check_rejects(R"({"prompt":"x","max_tokens":2.5})", "integer", "max_tokens");
    check_rejects(R"({"prompt":"x","max_tokens":18446744073709551615})", "integer", "max_tokens");
    check_rejects(R"({"prompt":"x","temperature":3})", "temperature", "temperature");
    check_rejects(R"({"prompt":"x","temperature":"hot"})", "number", "temperature");
    check_rejects(R"({"prompt":"x","top_p":0})", "top_p", "top_p");
    check_rejects(R"({"prompt":"x","top_k":-1})", "top_k", "top_k");
    check_rejects(R"({"prompt":"x","min_p":1.5})", "min_p", "min_p");
    check_rejects(R"({"prompt":"x","presence_penalty":2.5})", "presence_penalty",
                  "presence_penalty");
    check_rejects(R"({"prompt":"x","repeat_penalty":0})", "repeat_penalty", "repeat_penalty");
    check_rejects(R"({"prompt":"x","seed":-1})", "seed", "seed");
    check_rejects(R"({"prompt":"x","seed":4294967295})", "seed", "seed");
    check_rejects(R"({"prompt":"x","stop":["a","b","c","d","e"]})", "at most 4", "stop");
    check_rejects(R"({"prompt":"x","stop":[""]})", "non-empty", "stop");
    check_rejects(R"({"prompt":"x","stream":"yes"})", "boolean", "stream");
    check_rejects(R"({"prompt":"x","stream_options":true})", "object", "stream_options");
}

TEST_CASE("completion request: unsupported OpenAI features are rejected", "[request]") {
    check_rejects(R"({"prompt":"x","n":2})", "n = 1", "n");
    check_rejects(R"({"prompt":"x","best_of":3})", "best_of = 1", "best_of");
    check_rejects(R"({"prompt":"x","logprobs":true})", "not supported", "logprobs");
    check_rejects(R"({"prompt":"x","echo":true})", "not supported", "echo");
    check_rejects(R"({"prompt":"x","suffix":"z"})", "not supported", "suffix");
    check_rejects(R"({"prompt":"x","logit_bias":{"1":5}})", "not supported", "logit_bias");
    check_rejects(R"({"prompt":"x","tools":[{"type":"function"}]})", "not supported", "tools");
    check_rejects(R"({"prompt":"x","response_format":{"type":"json_schema"}})", "response_format",
                  "response_format");
    check_rejects(R"({"prompt":"x","response_format":"json"})", "response_format",
                  "response_format");
}

TEST_CASE("chat request: messages and content parts", "[request]") {
    const auto req = lisa::parse_chat_request(R"({"messages":[
        {"role":"developer","content":"be brief"},
        {"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"there"}]},
        {"role":"assistant","content":"hi","name":"bot"},
        {"role":"user","content":"again"}
    ]})",
                                              test_config());
    REQUIRE(req.messages.size() == 4);
    CHECK(req.messages[0].role == "system");
    CHECK(req.messages[0].content == "be brief");
    CHECK(req.messages[1].content == "hello there");
    CHECK(req.messages[2].role == "assistant");
    CHECK(req.params.max_tokens == 64);
}

TEST_CASE("chat request: invalid messages", "[request]") {
    check_rejects(R"({})", "messages", "messages", true);
    check_rejects(R"({"messages":[]})", "non-empty", "messages", true);
    check_rejects(R"({"messages":["hi"]})", "string role", "messages[0]", true);
    check_rejects(R"({"messages":[{"role":"tool","content":"x"}]})", "role", "messages[0].role",
                  true);
    check_rejects(R"({"messages":[{"role":"user"}]})", "required", "messages[0].content", true);
    check_rejects(R"({"messages":[{"role":"user","content":[{"type":"image_url"}]}]})", "text",
                  "messages[0].content", true);
    check_rejects(R"({"messages":[{"role":"user","content":"x"}],"top_p":2})", "top_p", "top_p",
                  true);
}
