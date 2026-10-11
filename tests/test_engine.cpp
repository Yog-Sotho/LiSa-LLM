// End-to-end tests against a real GGUF model. They run when LISA_TEST_MODEL points to a
// small model (CI uses ggml-org/models tinyllamas/stories260K.gguf) and skip otherwise.

#include "inference.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

lisa::InferenceEngine* shared_engine() {
    static const std::unique_ptr<lisa::InferenceEngine> engine = [] {
        std::unique_ptr<lisa::InferenceEngine> e;
        const char* path = std::getenv("LISA_TEST_MODEL");
        if (path != nullptr && *path != '\0') {
            lisa::Config cfg;
            cfg.model_path = path;
            cfg.n_threads = 2;
            cfg.max_context = 128;
            cfg.use_gpu = false;
            e = std::make_unique<lisa::InferenceEngine>(cfg);
        }
        return e;
    }();
    return engine.get();
}

lisa::CompletionRequest greedy(int max_new_tokens) {
    lisa::CompletionRequest req;
    req.prompt = "Once upon a time";
    req.max_new_tokens = max_new_tokens;
    req.temperature = 0.0f;
    req.top_p = 1.0f;
    return req;
}

std::string run(lisa::InferenceEngine& engine, const lisa::CompletionRequest& req,
                lisa::GenerationResult* result = nullptr) {
    std::string text;
    const auto r = engine.generate(engine.tokenize(req.prompt), req, [&](std::string_view piece) {
        text += piece;
        return true;
    });
    if (result != nullptr) {
        *result = r;
    }
    return text;
}

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
}

TEST_CASE("engine generates deterministic greedy text", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    lisa::GenerationResult result;
    const std::string first = run(*engine, greedy(16), &result);
    const std::string second = run(*engine, greedy(16));

    CHECK_FALSE(first.empty());
    CHECK(first == second);  // the KV cache is reset between requests
    CHECK(result.completion_tokens >= 1);
    CHECK(result.completion_tokens <= 16);
    CHECK(result.prompt_tokens >= 2);
}

TEST_CASE("engine honours max_new_tokens and cancellation", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto req = greedy(4);
    int calls = 0;
    const auto result = engine->generate(engine->tokenize(req.prompt), req,
                                         [&](std::string_view) { return ++calls < 2; });
    CHECK(result.reason == lisa::StopReason::cancelled);
    CHECK(calls == 2);

    lisa::GenerationResult limited;
    run(*engine, greedy(3), &limited);
    CHECK(limited.completion_tokens <= 3);
}

TEST_CASE("engine rejects requests larger than the context window", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto req = greedy(static_cast<int>(engine->info().n_ctx));
    CHECK_THROWS_AS(
        engine->generate(engine->tokenize(req.prompt), req, [](std::string_view) { return true; }),
        std::invalid_argument);
}

TEST_CASE("engine treats special-token text in prompts as plain text", "[engine]") {
    const lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    // "</s>" must not become the EOS control token.
    const auto plain = engine->tokenize("</s>");
    CHECK(plain.size() > 2);
}
