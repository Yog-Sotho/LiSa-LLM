#include "server.hpp"
#include "sandbox.hpp"
#include "model.hpp"
#include "inference.hpp"
#include "utils.hpp"
#include <signal.h>
#include <atomic>
#include <thread>
#include <sys/wait.h>

static std::atomic<bool> g_terminate{false};
static pid_t child_pid = 0;

void signal_handler(int) { g_terminate = true; }

int main(int argc, char* argv[]) {
    Config cfg = load_config("config.yaml");
    
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    
    Model model;
    model.load(cfg.model_path);
    log_info("Model loaded", {{"n_layer", model.hparams().n_layer},
                              {"n_embd", model.hparams().n_embd},
                              {"gpu", model.is_gpu()}});
    
    if (cfg.sandbox_enabled) {
        child_pid = fork();
        if (child_pid < 0) { perror("fork"); return 1; }
        if (child_pid == 0) {
            if (sandbox_init(cfg) != 0) _exit(1);
        } else {
            while (!g_terminate) {
                int status;
                pid_t w = waitpid(child_pid, &status, WNOHANG);
                if (w == child_pid) {
                    log_error("child exited", {{"status", status}});
                    return status;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            kill(child_pid, SIGTERM);
            waitpid(child_pid, nullptr, 0);
            return 0;
        }
    }
    
    InferenceEngine engine(model, cfg);
    HttpServer server(engine, cfg);
    server.start();
    
    return 0;
}
