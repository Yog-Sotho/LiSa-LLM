#pragma once

// A scripted Engine for HTTP tests that must control timing (queueing, draining): no
// model needed. Generation emits "x" pieces until max_tokens, and can be held open.

#include "engine.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <thread>

namespace lisa::test {

class FakeEngine final : public Engine {
public:
    // While held, generate() keeps emitting a piece every 10 ms (ignoring max_tokens)
    // until release() or until the callback asks it to stop.
    void hold() { held_ = true; }
    void release() {
        held_ = false;
        changed_.notify_all();
    }
    void set_ready(bool ready) { ready_ = ready; }

    // Blocks until `count` generations have started.
    void wait_for_generations(int count) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return started_ >= count; });
    }

    [[nodiscard]] std::vector<Token> tokenize(std::string_view text) override {
        return std::vector<Token>(std::max<std::size_t>(1, text.size()), 1);
    }
    [[nodiscard]] std::vector<Token> tokenize_chat(
        const std::vector<ChatMessage>& messages) override {
        const std::size_t size = std::accumulate(messages.begin(), messages.end(), std::size_t{1},
                                                 [](std::size_t sum, const ChatMessage& message) {
                                                     return sum + message.content.size();
                                                 });
        return std::vector<Token>(size, 1);
    }

    GenerationResult generate(const std::vector<Token>& prompt, const GenerationParams& params,
                              const PieceCallback& on_piece) override {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            ++started_;
        }
        changed_.notify_all();
        GenerationResult result;
        result.prompt_tokens = prompt.size();
        int emitted = 0;
        while (held_ || emitted < params.max_tokens) {
            if (!on_piece("x")) {
                result.reason = StopReason::cancelled;
                break;
            }
            ++emitted;
            if (held_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        result.completion_tokens = static_cast<std::size_t>(emitted);
        return result;
    }

    [[nodiscard]] ModelInfo info() const override {
        ModelInfo info;
        info.description = "fake";
        info.n_ctx = 4096;
        return info;
    }

    [[nodiscard]] EngineStatus status() const noexcept override {
        EngineStatus status;
        status.ready = ready_;
        return status;
    }

private:
    std::atomic<bool> held_{false};
    std::atomic<bool> ready_{true};
    std::mutex mutex_;
    std::condition_variable changed_;
    int started_ = 0;
};

}  // namespace lisa::test
