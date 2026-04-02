#include "utils.hpp"
#include "model_loader.cpp"
#include "inference.cpp"
#include "sandbox.cpp"
#include "cpp-httplib.h"
#include <signal.h>
#include <atomic>
#include <thread>

// ---------------------------------------------------------------------
// Global termination flag (set by signal handlers)
// ---------------------------------------------------------------------
static std::atomic<bool> g_terminate{false};

void handle_signal(int) { g_terminate = true; }

int main(int argc, char **argv) {
    // -------------------------------------------------
    // 1️⃣ Load runtime configuration
    // -------------------------------------------------
    Config cfg = load_config("config.yaml");

    // -------------------------------------------------
    // 2️⃣ Install signal handlers (SIGINT / SIGTERM)
    // -------------------------------------------------
    struct sigaction sa{};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // -------------------------------------------------
    // 3️⃣ Fork + sandbox (parent monitors child)
    // -------------------------------------------------
    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 1;
    }
    if (child == 0) {
        // ---- child: set up sandbox and continue as the server ----
        if (sandbox_init(cfg) != 0) _exit(1);
        // Child now runs the rest of the code (model load + HTTP server)
    } else {
        // ---- parent: monitor child, forward termination ----
        while (!g_terminate) {
            int status;
            pid_t w = waitpid(child, &status, WNOHANG);
            if (w == child) {
                std::cerr << "sandboxed child exited unexpectedly\n";
                return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        // Forward termination signal to child and wait
        kill(child, SIGTERM);
        waitpid(child, nullptr, 0);
        return 0;
    }

    // -------------------------------------------------
    // 4️⃣ Load the model (inside sandbox)
    // -------------------------------------------------
    Model model;
    try {
        model = load_model(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Model loading failed: " << e.what() << "\n";
        return 1;
    }

    // -------------------------------------------------
    // 5️⃣ Initialise inference engine
    // -------------------------------------------------
    InferenceEngine engine(model, cfg);

    // -------------------------------------------------
    // 6️⃣ HTTP server (cpp‑httplib)
    // -------------------------------------------------
    using namespace httplib;
    Server svr;

    // Health‑check endpoint
    svr.Get("/healthz", [&](const Request&, Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Completion endpoint – streams tokens as a JSON array (chunked)
    svr.Post("/v1/completions", [&](const Request& req, Response& res) {
        json payload;
        try {
            payload = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }

        std::string prompt = payload.value("prompt", "");
        int max_new = payload.value("max_new_tokens", 64);
        float temperature = payload.value("temperature", 0.8f);
        float top_p = payload.value("top_p", 0.95f);

        std::vector<std::string> tokens;
        try {
            tokens = engine.generate(prompt, max_new, temperature, top_p);
        } catch (const std::exception& e) {
            res.status = 500;
            json err = { {"error", e.what()} };
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Chunked JSON stream – each chunk is a single token object
        res.set_header("Transfer-Encoding", "chunked");
        res.set_content_provider(
            "application/json",
            [tokens, idx = size_t{0}](size_t offset, DataSink& sink) mutable {
                if (idx >= tokens.size()) return false; // EOF
                json chunk = { {"token", tokens[idx++]} };
                sink.write(chunk.dump());
                return true;
            });
    });

    // -------------------------------------------------
    // 7️⃣ Run the server (blocking)
    // -------------------------------------------------
    std::cout << "[" << iso8601_now() << "] Listening on "
              << cfg.listen_addr << ":" << cfg.listen_port << "\n";

    svr.listen(cfg.listen_addr.c_str(), cfg.listen_port);

    // -------------------------------------------------
    // 8️⃣ Cleanup
    // -------------------------------------------------
    unload_model(model);
    return 0;
}
