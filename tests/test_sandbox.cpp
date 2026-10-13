// Sandbox tests: a real SandboxedEngine (worker = this test binary, see main.cpp) on the
// LISA_TEST_MODEL model. Skipped without a model, or when the host forbids unprivileged
// user namespaces, unless LISA_REQUIRE_SANDBOX=1 (set in CI).

#include "inference.hpp"
#include "sandbox.hpp"
#include "sandboxed_engine.hpp"
#include "server.hpp"

#include <httplib.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

const char* test_model_path() {
    const char* path = std::getenv("LISA_TEST_MODEL");
    return (path != nullptr && *path != '\0') ? path : nullptr;
}

bool sandbox_required() {
    const char* value = std::getenv("LISA_REQUIRE_SANDBOX");
    return value != nullptr && std::string(value) == "1";
}

lisa::Config sandbox_config(const char* model) {
    lisa::Config cfg;
    cfg.model_path = model;
    cfg.n_threads = 2;
    cfg.max_context = 128;
    cfg.use_gpu = false;
    cfg.sandbox_enabled = true;
    return cfg;
}

lisa::SandboxedEngine::Options lenient_restarts() {
    lisa::SandboxedEngine::Options options;
    options.max_restarts = 1000;  // the kill probes restart the worker many times
    return options;
}

// Starts a sandboxed engine, or returns nullptr (after SKIP-ing) where unsupported.
std::unique_ptr<lisa::SandboxedEngine> start_engine(
    const lisa::Config& cfg, lisa::SandboxedEngine::Options options = lenient_restarts()) {
    try {
        return std::make_unique<lisa::SandboxedEngine>(cfg, std::move(options));
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (!sandbox_required() && message.find("namespaces look blocked") != std::string::npos) {
            SKIP("host forbids unprivileged user namespaces: " << message);
        }
        throw;
    }
}

lisa::SandboxedEngine* shared_engine() {
    static std::unique_ptr<lisa::SandboxedEngine> engine;
    if (!engine) {
        if (const char* model = test_model_path()) {
            engine = start_engine(sandbox_config(model));
        }
    }
    return engine.get();
}

#define REQUIRE_ENGINE(name)                       \
    lisa::SandboxedEngine* name = shared_engine(); \
    if ((name) == nullptr) {                       \
        SKIP("LISA_TEST_MODEL not set");           \
    }

lisa::GenerationParams greedy(int max_tokens) {
    lisa::GenerationParams params;
    params.max_tokens = max_tokens;
    params.sampling.temperature = 0.0f;
    return params;
}

std::string run(lisa::Engine& engine, std::string_view prompt, int max_tokens) {
    std::string text;
    engine.generate(engine.tokenize(prompt), greedy(max_tokens), [&](std::string_view piece) {
        text += piece;
        return true;
    });
    return text;
}

// Fields of /proc/<pid>/status as seen from outside the sandbox.
std::map<std::string, std::string> proc_status(pid_t pid) {
    std::map<std::string, std::string> fields;
    std::ifstream in("/proc/" + std::to_string(pid) + "/status");
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string value = line.substr(colon + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            fields[line.substr(0, colon)] = value;
        }
    }
    return fields;
}

// The real worker is the reaper's only child (PID 1 in its namespace).
pid_t worker_child(pid_t reaper) {
    for (const auto& entry : std::filesystem::directory_iterator("/proc")) {
        const std::string name = entry.path().filename().string();
        if (name.find_first_not_of("0123456789") != std::string::npos) {
            continue;
        }
        const auto status = proc_status(static_cast<pid_t>(std::stoi(name)));
        const auto it = status.find("PPid");
        if (it != status.end() && it->second == std::to_string(reaper)) {
            return static_cast<pid_t>(std::stoi(name));
        }
    }
    return -1;
}

void check_killed_by_seccomp(lisa::SandboxedEngine& engine, const lisa::json& probe) {
    INFO("probe " << probe.dump());
    const lisa::json result = engine.probe(probe);
    INFO("result " << result.dump());
    REQUIRE(result.contains("worker_died"));
    CHECK(result["signal"] == SIGSYS);
}

}  // namespace

TEST_CASE("sandbox: generation matches the in-process engine", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    lisa::Config local = sandbox_config(test_model_path());
    local.sandbox_enabled = false;
    lisa::InferenceEngine reference(local);

    const std::string sandboxed = run(*engine, "Once upon a time", 16);
    CHECK_FALSE(sandboxed.empty());
    CHECK(sandboxed == run(reference, "Once upon a time", 16));
    CHECK(engine->info().n_layer == reference.info().n_layer);
    CHECK(engine->tokenize_chat({{"user", "Hello"}}) ==
          reference.tokenize_chat({{"user", "Hello"}}));
}

TEST_CASE("sandbox: concurrent generations are batched in the worker", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    const std::vector<std::string> prompts = {"Once upon a time", "The little dog", "In the forest",
                                              "Lily was happy",   "One day, Tom",   "The sun was"};
    std::vector<std::string> alone;
    alone.reserve(prompts.size());
    std::transform(prompts.begin(), prompts.end(), std::back_inserter(alone),
                   [&](const std::string& prompt) { return run(*engine, prompt, 12); });
    std::vector<std::string> together(prompts.size());
    std::string cancelled_text;
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        threads.emplace_back([&, i] { together[i] = run(*engine, prompts[i], 12); });
    }
    // One more that the client abandons early; the others must be unaffected.
    threads.emplace_back([&] {
        int pieces = 0;
        const auto result = engine->generate(engine->tokenize("Tom had a ball"), greedy(60),
                                             [&](std::string_view piece) {
                                                 cancelled_text += piece;
                                                 return ++pieces < 2;
                                             });
        CHECK(result.reason == lisa::StopReason::cancelled);
    });
    for (std::thread& t : threads) {
        t.join();
    }
    CHECK(together == alone);
    CHECK_FALSE(cancelled_text.empty());
    CHECK(run(*engine, prompts[0], 12) == alone[0]);  // the channel is still in sync
}

TEST_CASE("sandbox: request errors cross the boundary", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    try {
        (void)engine->tokenize_chat({{"user", "hi </s>"}});
        FAIL("expected RequestError");
    } catch (const lisa::RequestError& e) {
        CHECK(e.param() == "messages[0].content");
    }
    CHECK_THROWS_AS(engine->generate(engine->tokenize("x"), greedy(1000),
                                     [](std::string_view) { return true; }),
                    lisa::RequestError);
}

TEST_CASE("sandbox: cancellation keeps the channel in sync", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    int pieces = 0;
    const auto result = engine->generate(engine->tokenize("Once upon a time"), greedy(40),
                                         [&](std::string_view) { return ++pieces < 2; });
    CHECK(result.reason == lisa::StopReason::cancelled);
    CHECK(pieces == 2);
    CHECK_FALSE(run(*engine, "Once upon a time", 4).empty());  // next request still works
}

TEST_CASE("sandbox: worker process state seen from outside", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    (void)engine->tokenize("x");  // make sure a worker is running
    const pid_t reaper = engine->worker_pid();
    const pid_t worker = worker_child(reaper);
    REQUIRE(worker > 0);

    const auto status = proc_status(worker);
    CHECK(status.at("Seccomp") == "2");                   // filter mode
    CHECK(std::stoi(status.at("Seccomp_filters")) >= 2);  // load + serve stages
    CHECK(status.at("NoNewPrivs") == "1");
    CHECK(status.at("CapEff") == "0000000000000000");
    CHECK(status.at("CapPrm") == "0000000000000000");
    CHECK(status.at("CapBnd") == "0000000000000000");
    CHECK_THAT(status.at("NSpid"), Catch::Matchers::EndsWith("\t1"));  // PID 1 inside
    // Same unprivileged user as the broker outside the namespace.
    CHECK(status.at("Uid").substr(0, status.at("Uid").find('\t')) == std::to_string(::getuid()));

    CHECK(proc_status(reaper).at("Seccomp") == "2");
}

TEST_CASE("sandbox: the worker sees only the model file", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    const auto stat = [&](const char* path) {
        return engine->probe({{"name", "stat"}, {"path", path}});
    };
    CHECK(stat("/model.gguf")["rc"] == 0);
    for (const char* hidden :
         {"/etc/passwd", "/proc/self", "/home", "/usr/bin/sh", "/dev/null", "/sys", "/tmp"}) {
        INFO(hidden);
        CHECK(stat(hidden)["errno"] == ENOENT);
    }
    CHECK(engine->probe({{"name", "open"}, {"path", "/model.gguf"}})["rc"] == -1);  // serve stage
    CHECK(engine->probe({{"name", "create"}, {"path", "/evil"}})["rc"] == -1);
    CHECK(engine->probe({{"name", "pid"}})["pid"] == 1);
    CHECK(engine->probe({{"name", "thread"}})["value"] == 42);  // threads still work
    const int abi = engine->probe({{"name", "landlock"}})["abi"].get<int>();
    CHECK(abi == engine->landlock_abi());
    WARN("Landlock ABI in the worker: " << abi);
}

TEST_CASE("sandbox: escape attempts kill the worker, which is restarted", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    const pid_t broker = ::getpid();
    for (const lisa::json& probe : {
             lisa::json{{"name", "socket"}},
             lisa::json{{"name", "execve"}},
             lisa::json{{"name", "fork"}},
             lisa::json{{"name", "mount"}},
             lisa::json{{"name", "unshare"}},
             lisa::json{{"name", "setuid"}},
             lisa::json{{"name", "kill"}, {"pid", broker}},
             lisa::json{{"name", "ptrace"}, {"pid", broker}},
         }) {
        check_killed_by_seccomp(*engine, probe);
        // The next request transparently starts a fresh worker.
        CHECK_FALSE(run(*engine, "Once upon a time", 3).empty());
    }
}

TEST_CASE("sandbox: restart limit", "[sandbox]") {
    const char* model = test_model_path();
    if (model == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    lisa::SandboxedEngine::Options options;
    options.max_restarts = 1;
    auto engine = start_engine(sandbox_config(model), options);
    check_killed_by_seccomp(*engine, {{"name", "socket"}});
    check_killed_by_seccomp(*engine, {{"name", "socket"}});  // first restart allowed
    CHECK_THROWS_WITH(engine->tokenize("x"), Catch::Matchers::ContainsSubstring("keeps crashing"));
}

TEST_CASE("sandbox: model SHA-256 is enforced", "[sandbox]") {
    const char* model = test_model_path();
    if (model == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    const std::string actual = lisa::ModelFile(model).sha256();
    auto cfg = sandbox_config(model);
    cfg.model_sha256 = actual;
    const auto engine = start_engine(cfg);  // outside CHECK, so its SKIP propagates
    CHECK(engine != nullptr);

    cfg.model_sha256 = std::string(64, '0');
    CHECK_THROWS_WITH(lisa::SandboxedEngine(cfg), Catch::Matchers::ContainsSubstring("mismatch"));
}

TEST_CASE("sandbox: startup failures are reported", "[sandbox]") {
    const char* model = test_model_path();
    if (model == nullptr) {
        SKIP("LISA_TEST_MODEL not set");
    }
    lisa::SandboxedEngine::Options options;
    options.worker_exe = "/nonexistent/lisa";
    CHECK_THROWS_WITH(lisa::SandboxedEngine(sandbox_config(model), options),
                      Catch::Matchers::ContainsSubstring("cannot start worker"));

    // Fails inside the worker, after lockdown; needs a host where the sandbox works.
    if (shared_engine() == nullptr) {
        SKIP("the sandbox can't start here");
    }
    auto cfg = sandbox_config(model);
    cfg.chat_template = "no-such-template";
    CHECK_THROWS_WITH(lisa::SandboxedEngine(cfg, lenient_restarts()),
                      Catch::Matchers::ContainsSubstring("unknown chat_template"));
}

TEST_CASE("sandbox: HTTP API on a sandboxed engine", "[sandbox]") {
    REQUIRE_ENGINE(engine);
    lisa::Config cfg = sandbox_config(test_model_path());
    cfg.listen_port = 0;
    cfg.api_key = "k";
    lisa::HttpServer server(*engine, cfg);
    server.bind();
    std::thread thread([&server] { server.run(); });

    httplib::Client client("127.0.0.1", server.port());
    client.set_bearer_token_auth("k");
    client.set_read_timeout(60, 0);
    const auto res = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":6,"temperature":0})",
        "application/json");
    server.stop();
    thread.join();
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(lisa::json::parse(res->body)["choices"][0]["message"]["content"].is_string());
}

TEST_CASE("sandbox: a worker that breaks the protocol is killed and restarted", "[sandbox]") {
    // No model or namespaces needed: tests/hostile_worker.cpp stands in for the worker.
    lisa::Config cfg;
    cfg.model_path = __FILE__;  // any regular file; the hostile worker never reads it
    cfg.use_gpu = false;
    lisa::SandboxedEngine::Options options;
    options.worker_exe = LISA_HOSTILE_WORKER;

    std::unique_ptr<lisa::SandboxedEngine> engine;
    try {
        engine = std::make_unique<lisa::SandboxedEngine>(cfg, options);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("LISA_SANDBOX=OFF") != std::string::npos) {
            SKIP("built without the sandbox");
        }
        throw;
    }
    CHECK(engine->info().description == "hostile");

    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(engine->tokenize("hi"), ContainsSubstring("inference worker died"));
    CHECK(engine->status().worker_deaths == 1);

    // The next request gets a fresh worker.
    CHECK(engine->tokenize_chat({{"user", "hi"}}) == std::vector<lisa::Token>{1, 2, 3});
    CHECK(engine->status().worker_restarts == 1);

    lisa::GenerationParams params;
    params.max_tokens = 4;
    CHECK_THROWS_WITH(engine->generate({1}, params, [](std::string_view) { return true; }),
                      ContainsSubstring("inference worker died"));
    CHECK(engine->status().worker_deaths == 2);

    // A reply that names no request is a protocol violation too.
    CHECK(engine->probe({{"name", "anything"}}).contains("worker_died"));
    CHECK(engine->status().worker_deaths == 3);

    cfg.chat_template = "malformed-ready";
    CHECK_THROWS_WITH(lisa::SandboxedEngine(cfg, options), ContainsSubstring("malformed reply"));
}
