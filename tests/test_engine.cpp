// End-to-end tests against a real GGUF model. They run when LISA_TEST_MODEL points to a
// small model (CI uses ggml-org/models tinyllamas/stories260K.gguf) and skip otherwise.

#include "inference.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

TEST_CASE("engine: concurrent requests don't affect each other", "[engine][batching]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    REQUIRE(engine->slots() == 4);
    // More requests than slots, with different prompts and lengths: each must produce
    // exactly what it produces alone.
    const std::vector<std::string> prompts = {"Once upon a time", "The little dog", "In the forest",
                                              "Lily was happy",   "One day, Tom",   "The sun was"};
    std::vector<std::string> alone;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        alone.push_back(
            run(*engine, engine->tokenize(prompts[i]), greedy(8 + static_cast<int>(i) * 3)));
    }
    for (int round = 0; round < 3; ++round) {
        std::vector<std::string> together(prompts.size());
        std::vector<std::thread> threads;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            threads.emplace_back([&, i] {
                together[i] =
                    run(*engine, engine->tokenize(prompts[i]), greedy(8 + static_cast<int>(i) * 3));
            });
        }
        for (std::thread& t : threads) {
            t.join();
        }
        CHECK(together == alone);
    }
}

TEST_CASE("engine: the prompt cache reuses a shared prefix", "[engine][batching]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto prompt = engine->tokenize("Once upon a time, there was a little girl named");
    lisa::GenerationResult first;
    lisa::GenerationResult second;
    const std::string a = run(*engine, prompt, greedy(10), &first);
    const std::string b = run(*engine, prompt, greedy(10), &second);
    CHECK(a == b);
    CHECK(second.cached_tokens == prompt.size() - 1);

    // A longer prompt starting with the same text reuses that part.
    auto extended = prompt;
    const auto more = engine->tokenize(" Lily. She");
    extended.insert(extended.end(), more.begin() + 1, more.end());  // skip BOS
    lisa::GenerationResult third;
    (void)run(*engine, extended, greedy(4), &third);
    CHECK(third.cached_tokens >= prompt.size() - 1);
}

TEST_CASE("engine: cancelling one request leaves the others running", "[engine][batching]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const auto prompt = engine->tokenize("The little dog");
    const std::string expected = run(*engine, prompt, greedy(30));

    std::string survivor;
    lisa::GenerationResult cancelled_result;
    std::thread cancelled([&] {
        int pieces = 0;
        cancelled_result = engine->generate(engine->tokenize("In the forest"), greedy(60),
                                            [&](std::string_view) { return ++pieces < 3; });
    });
    std::thread kept([&] { survivor = run(*engine, prompt, greedy(30)); });
    cancelled.join();
    kept.join();
    CHECK(cancelled_result.reason == lisa::StopReason::cancelled);
    CHECK(survivor == expected);
}

TEST_CASE("engine: async submit and cancel before start", "[engine][batching]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    std::mutex mutex;
    std::condition_variable done;
    int finished = 0;
    std::vector<lisa::StopReason> reasons;
    lisa::InferenceEngine::Callbacks callbacks;
    callbacks.on_piece = [](std::string_view) { return true; };
    callbacks.on_done = [&](const lisa::GenerationResult& r) {
        const std::lock_guard<std::mutex> lock(mutex);
        reasons.push_back(r.reason);
        ++finished;
        done.notify_all();
    };
    callbacks.on_error = [&](const std::exception_ptr&) {
        const std::lock_guard<std::mutex> lock(mutex);
        ++finished;
        done.notify_all();
    };
    // Fill every slot with long requests, then queue one more and cancel it at once.
    std::vector<lisa::InferenceEngine::RequestId> ids;
    for (std::size_t i = 0; i <= engine->slots(); ++i) {
        ids.push_back(engine->submit(engine->tokenize("Once upon a time"), greedy(100), callbacks));
    }
    engine->cancel(ids.back());
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
        engine->cancel(ids[i]);
    }
    std::unique_lock<std::mutex> lock(mutex);
    done.wait(lock, [&] { return finished == static_cast<int>(ids.size()); });
    CHECK(std::count(reasons.begin(), reasons.end(), lisa::StopReason::cancelled) >= 1);
}

TEST_CASE("engine: a busy slot's cached prefix is copied to a new slot", "[engine][batching]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    // Long enough (> 32 shared tokens) for a copy to be worth it.
    const auto prompt = engine->tokenize(
        "Once upon a time, there was a little girl named Lily. She loved to play outside in "
        "the sun with her dog. One day, she saw a big red ball in the park and");
    REQUIRE(prompt.size() > 40);
    const std::string expected = run(*engine, prompt, greedy(8));

    // Keep the slot holding the prompt busy with a long generation...
    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool finished = false;
    lisa::InferenceEngine::Callbacks callbacks;
    callbacks.on_piece = [&](std::string_view) {
        const std::lock_guard<std::mutex> lock(mutex);
        started = true;
        changed.notify_all();
        return true;
    };
    callbacks.on_done = [&](const lisa::GenerationResult&) {
        const std::lock_guard<std::mutex> lock(mutex);
        finished = true;
        changed.notify_all();
    };
    callbacks.on_error = [&](const std::exception_ptr&) {
        const std::lock_guard<std::mutex> lock(mutex);
        finished = true;
        changed.notify_all();
    };
    const auto busy = engine->submit(prompt, greedy(60), callbacks);
    {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return started; });
    }
    // ...so the same prompt goes to another slot, which copies the prefix.
    lisa::GenerationResult result;
    CHECK(run(*engine, prompt, greedy(8), &result) == expected);
    CHECK(result.cached_tokens == prompt.size() - 1);

    engine->cancel(busy);
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] { return finished; });
}

TEST_CASE("engine: JSON mode only produces JSON", "[engine]") {
    lisa::InferenceEngine* engine = shared_engine();
    if (engine == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    for (const float temperature : {0.0f, 0.9f}) {
        lisa::GenerationParams params = greedy(80);
        params.json_object = true;
        params.sampling.temperature = temperature;
        params.sampling.seed = 7;
        lisa::GenerationResult result;
        const std::string text =
            run(*engine, engine->tokenize("Once upon a time"), params, &result);
        INFO(text);
        REQUIRE_FALSE(text.empty());
        CHECK(text.front() == '{');
        if (result.reason == lisa::StopReason::eos) {
            // The grammar only allows the end once the object is complete.
            CHECK_NOTHROW(lisa::json::parse(text));
        } else {
            // Cut off by max_tokens: what there is must still be valid so far.
            CHECK(lisa::json::accept(text) == false);
            CHECK(result.reason == lisa::StopReason::length);
        }
    }
}
