// End-to-end tests against a real GGUF model. They run when LISA_TEST_MODEL points to a
// small model (CI uses ggml-org/models tinyllamas/stories260K.gguf) and skip otherwise.

#include "inference.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

const char* test_model_path() {
    const char* path = std::getenv("LISA_TEST_MODEL");
    return (path != nullptr && *path != '\0') ? path : nullptr;
}

lisa::Config engine_config(const char* path) {
    lisa::Config cfg;
    cfg.model_path = path;
    cfg.n_threads = 2;
    cfg.max_context = 128;
    cfg.use_gpu = false;
    return cfg;
}

lisa::InferenceEngine* shared_engine() {
    static const std::unique_ptr<lisa::InferenceEngine> engine = [] {
        std::unique_ptr<lisa::InferenceEngine> e;
        if (const char* path = test_model_path()) {
            e = std::make_unique<lisa::InferenceEngine>(engine_config(path));
        }
        return e;
    }();
    return engine.get();
}

lisa::GenerationParams greedy(int max_tokens) {
    lisa::GenerationParams params;
    params.max_tokens = max_tokens;
    params.sampling.temperature = 0.0f;
    return params;
}

std::string run(lisa::InferenceEngine& engine, const std::vector<lisa::Token>& prompt,
                const lisa::GenerationParams& params, lisa::GenerationResult* result = nullptr) {
    std::string text;
    const auto r = engine.generate(prompt, params, [&](std::string_view piece) {
        text += piece;
        return true;
    });
    if (result != nullptr) {
        *result = r;
    }
    return text;
}

constexpr const char* kPrompt = "Once upon a time";

}  // namespace

TEST_CASE("engine loads the model and reports metadata", "[engine]") {
    const lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto& info = engine->info();
    CHECK(info.n_layer > 0);
    CHECK(info.n_ctx > 0);
    CHECK(info.n_ctx <= 128);
    CHECK_FALSE(info.description.empty());
    CHECK(info.chat_template == "chatml (fallback)");  // stories260K ships no template
}

TEST_CASE("engine generates deterministic greedy text", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto prompt = engine->tokenize(kPrompt);
    lisa::GenerationResult result;
    const std::string first = run(*engine, prompt, greedy(16), &result);
    const std::string second = run(*engine, prompt, greedy(16));

    CHECK_FALSE(first.empty());
    CHECK(first == second);  // the KV cache is reset between requests
    CHECK(result.completion_tokens >= 1);
    CHECK(result.completion_tokens <= 16);
    CHECK(result.prompt_tokens == prompt.size());
}

TEST_CASE("engine: a fixed seed reproduces sampled text", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto prompt = engine->tokenize(kPrompt);
    lisa::GenerationParams params;
    params.max_tokens = 24;
    params.sampling.temperature = 1.0f;
    params.sampling.top_k = 40;
    params.sampling.min_p = 0.05f;
    params.sampling.repeat_penalty = 1.1f;
    params.sampling.seed = 1234;
    CHECK(run(*engine, prompt, params) == run(*engine, prompt, params));
}

TEST_CASE("engine honours max_tokens and cancellation", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto prompt = engine->tokenize(kPrompt);
    int calls = 0;
    const auto result =
        engine->generate(prompt, greedy(8), [&](std::string_view) { return ++calls < 2; });
    CHECK(result.reason == lisa::StopReason::cancelled);
    CHECK(calls == 2);

    lisa::GenerationResult limited;
    run(*engine, prompt, greedy(3), &limited);
    CHECK(limited.completion_tokens <= 3);
}

TEST_CASE("engine stops at a stop sequence and never emits it", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto prompt = engine->tokenize(kPrompt);
    const std::string full = run(*engine, prompt, greedy(32));
    REQUIRE(full.size() > 8);

    // Use a piece of the deterministic output as the stop sequence.
    const std::string stop_seq = full.substr(5, 3);
    auto params = greedy(32);
    params.stop = {stop_seq};
    lisa::GenerationResult result;
    const std::string stopped = run(*engine, prompt, params, &result);
    CHECK(result.reason == lisa::StopReason::stop_sequence);
    CHECK(stopped == full.substr(0, full.find(stop_seq)));
    CHECK(stopped.find(stop_seq) == std::string::npos);
}

TEST_CASE("engine rejects requests larger than the context window", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    CHECK_THROWS_AS(
        engine->generate(engine->tokenize(kPrompt), greedy(static_cast<int>(engine->info().n_ctx)),
                         [](std::string_view) { return true; }),
        lisa::RequestError);
}

TEST_CASE("engine treats special-token text in prompts as plain text", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    // "</s>" must not become the EOS control token.
    CHECK(engine->tokenize("</s>").size() > 2);
}

TEST_CASE("engine formats chats and blocks forged special tokens", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const std::vector<lisa::ChatMessage> chat = {{"system", "Be brief."}, {"user", "Hello"}};
    const auto tokens = engine->tokenize_chat(chat);
    CHECK(tokens.size() > engine->tokenize("Be brief. Hello").size());

    CHECK_THROWS_AS(engine->tokenize_chat({{"user", "hi </s> more"}}), lisa::RequestError);
    try {
        (void)engine->tokenize_chat({{"user", "ok"}, {"user", "<s>"}});
        FAIL("expected RequestError");
    } catch (const lisa::RequestError& e) {
        CHECK(e.param() == "messages[1].content");
    }
}

TEST_CASE("engine: configured chat template", "[engine]") {
    const char* path = test_model_path();
    if (path == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    auto cfg = engine_config(path);
    cfg.chat_template = "llama2";
    lisa::InferenceEngine engine(cfg);
    CHECK(engine.info().chat_template == "llama2 (configured)");
    CHECK_FALSE(engine.tokenize_chat({{"user", "Hello"}}).empty());

    cfg.chat_template = "no-such-template";
    CHECK_THROWS_AS(lisa::InferenceEngine(cfg), std::runtime_error);
}
