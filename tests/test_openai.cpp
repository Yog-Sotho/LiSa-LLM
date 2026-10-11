#include "openai.hpp"

#include <catch2/catch_test_macros.hpp>

namespace {

lisa::GenerationResult result(lisa::StopReason reason) {
    lisa::GenerationResult r;
    r.prompt_tokens = 5;
    r.completion_tokens = 3;
    r.reason = reason;
    return r;
}

}  // namespace

TEST_CASE("make_meta creates unique prefixed ids", "[openai]") {
    const auto a = lisa::openai::make_meta("chatcmpl-", "m");
    const auto b = lisa::openai::make_meta("chatcmpl-", "m");
    CHECK(a.id.starts_with("chatcmpl-"));
    CHECK(a.id.size() == 9 + 24);
    CHECK(a.id != b.id);
    CHECK(a.created > 1'700'000'000);
    CHECK(a.model == "m");
}

TEST_CASE("finish reasons and usage", "[openai]") {
    using lisa::StopReason;
    CHECK(lisa::openai::finish_reason(StopReason::eos) == "stop");
    CHECK(lisa::openai::finish_reason(StopReason::stop_sequence) == "stop");
    CHECK(lisa::openai::finish_reason(StopReason::length) == "length");
    CHECK(lisa::openai::finish_reason(StopReason::cancelled).is_null());

    const auto usage = lisa::openai::usage(result(StopReason::eos));
    CHECK(usage["prompt_tokens"] == 5);
    CHECK(usage["completion_tokens"] == 3);
    CHECK(usage["total_tokens"] == 8);
}

TEST_CASE("chat completion bodies follow the OpenAI shape", "[openai]") {
    const auto meta = lisa::openai::make_meta("chatcmpl-", "m");
    const auto body =
        lisa::openai::chat_completion(meta, "hello", result(lisa::StopReason::length));
    CHECK(body["object"] == "chat.completion");
    CHECK(body["model"] == "m");
    REQUIRE(body["choices"].size() == 1);
    CHECK(body["choices"][0]["index"] == 0);
    CHECK(body["choices"][0]["message"]["role"] == "assistant");
    CHECK(body["choices"][0]["message"]["content"] == "hello");
    CHECK(body["choices"][0]["finish_reason"] == "length");
    CHECK(body["usage"]["total_tokens"] == 8);

    const auto chunk = lisa::openai::chat_chunk(meta, {{"content", "he"}});
    CHECK(chunk["object"] == "chat.completion.chunk");
    CHECK(chunk["id"] == meta.id);
    CHECK(chunk["choices"][0]["delta"]["content"] == "he");
    CHECK(chunk["choices"][0]["finish_reason"].is_null());
    CHECK_FALSE(chunk.contains("usage"));

    const auto usage =
        lisa::openai::usage_chunk(meta, "chat.completion.chunk", result(lisa::StopReason::eos));
    CHECK(usage["choices"].empty());
    CHECK(usage["usage"]["completion_tokens"] == 3);
}

TEST_CASE("text completion bodies follow the OpenAI shape", "[openai]") {
    const auto meta = lisa::openai::make_meta("cmpl-", "m");
    const auto body = lisa::openai::completion(meta, "abc", result(lisa::StopReason::eos));
    CHECK(body["object"] == "text_completion");
    CHECK(body["choices"][0]["text"] == "abc");
    CHECK(body["choices"][0]["finish_reason"] == "stop");
    CHECK(body["choices"][0]["logprobs"].is_null());

    const auto chunk = lisa::openai::completion_chunk(meta, "", "stop");
    CHECK(chunk["choices"][0]["finish_reason"] == "stop");
}

TEST_CASE("models and error bodies", "[openai]") {
    const auto models = lisa::openai::models("llama", 123);
    CHECK(models["object"] == "list");
    CHECK(models["data"][0]["id"] == "llama");
    CHECK(models["data"][0]["object"] == "model");
    CHECK(models["data"][0]["created"] == 123);

    const auto err = lisa::openai::error("bad", "invalid_request_error", "top_p");
    CHECK(err["error"]["message"] == "bad");
    CHECK(err["error"]["type"] == "invalid_request_error");
    CHECK(err["error"]["param"] == "top_p");
    CHECK(err["error"]["code"].is_null());
}
