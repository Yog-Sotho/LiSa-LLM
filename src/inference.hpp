#pragma once

#include "engine.hpp"
#include "request.hpp"
#include "utils.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace lisa {

// Text generation on top of libllama (llama.cpp) with continuous batching: up to
// cfg.parallel_slots requests generate together. A scheduler thread owns the llama
// context; each step evaluates one batch holding the next token of every generating
// request plus chunks of newly admitted prompts. A new request takes the free slot whose
// cached tokens share the longest prefix with its prompt and only evaluates the rest.
class InferenceEngine final : public Engine {
public:
    // Callbacks of an asynchronous generation. They run on the scheduler thread, so they
    // must not block for long.
    struct Callbacks {
        std::function<bool(std::string_view piece)> on_piece;  // false = cancel
        std::function<void(const GenerationResult&)> on_done;
        std::function<void(std::exception_ptr)> on_error;
    };
    using RequestId = std::uint64_t;

    // Loads the GGUF model in cfg.model_path and selects the chat template
    // (cfg.chat_template, else the model's own, else ChatML). Throws std::runtime_error.
    explicit InferenceEngine(const Config& cfg);
    ~InferenceEngine() override;  // fails requests still in progress

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = delete;
    InferenceEngine& operator=(InferenceEngine&&) = delete;

    // Tokenises a raw completion prompt. Special-token text (e.g. "</s>") is treated as
    // plain text. Thread-safe. Throws RequestError.
    [[nodiscard]] std::vector<Token> tokenize(std::string_view text) override;

    // Formats `messages` with the chat template and tokenises the result. Message
    // contents that contain special-token text are rejected, so a user cannot forge
    // role boundaries. Thread-safe. Throws RequestError.
    [[nodiscard]] std::vector<Token> tokenize_chat(
        const std::vector<ChatMessage>& messages) override;

    // Generates a completion, calling `on_piece` on this thread. Thread-safe: concurrent
    // calls are batched. Throws RequestError when the request does not fit the context
    // window, std::runtime_error on engine failure.
    GenerationResult generate(const std::vector<Token>& prompt_tokens,
                              const GenerationParams& params,
                              const PieceCallback& on_piece) override;

    // Queues a generation and returns at once; exactly one of on_done/on_error is called
    // later. Throws RequestError for requests that can never fit. Thread-safe.
    RequestId submit(std::vector<Token> prompt_tokens, GenerationParams params,
                     Callbacks callbacks);

    // Stops a queued or running generation; its on_done reports StopReason::cancelled.
    void cancel(RequestId id);

    [[nodiscard]] ModelInfo info() const override { return info_; }
    [[nodiscard]] std::size_t slots() const noexcept { return slots_.size(); }

    // Runs one tiny generation so thread pools and lazily initialised state exist before
    // the sandbox narrows the allowed system calls.
    void warm_up();

private:
    struct ModelDeleter {
        void operator()(llama_model* model) const noexcept;
    };
    struct ContextDeleter {
        void operator()(llama_context* ctx) const noexcept;
    };
    class Batch;
    struct Job;
    struct Slot;

    [[nodiscard]] std::vector<Token> tokenize_text(std::string_view text, bool add_special,
                                                   bool parse_special) const;
    [[nodiscard]] std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;
    [[nodiscard]] std::string token_to_piece(Token token) const;
    void select_chat_template(const std::string& configured);

    // Scheduler (runs on scheduler_ only, except where noted).
    void run_scheduler();
    // Called with mutex_ held. Returns requests cancelled while queued; the caller reports
    // them after unlocking (callbacks never run under mutex_).
    [[nodiscard]] std::vector<std::shared_ptr<Job>> admit_pending_locked();
    void start_job(Slot& slot, std::shared_ptr<Job> job);
    void step();
    void accept_token(Slot& slot, std::int32_t batch_index);
    void finish(Slot& slot, StopReason reason, bool flush);
    void fail_active(const std::exception_ptr& error);

    std::unique_ptr<llama_model, ModelDeleter> model_;
    std::unique_ptr<llama_context, ContextDeleter> ctx_;
    const llama_vocab* vocab_ = nullptr;
    std::string chat_template_;  // template text or built-in name for llama_chat_apply_template
    ModelInfo info_;
    std::uint32_t n_batch_ = 0;
    bool can_copy_sequences_ = false;  // prefix copies between slots are possible

    std::unique_ptr<Batch> batch_;              // scheduler thread only
    std::vector<std::unique_ptr<Slot>> slots_;  // scheduler thread only
    std::uint64_t use_counter_ = 0;             // scheduler thread only (LRU order)

    std::mutex mutex_;  // guards the members below
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> pending_;
    std::unordered_map<RequestId, std::shared_ptr<Job>> jobs_;  // pending and running
    RequestId next_id_ = 1;
    bool stopping_ = false;
    std::thread scheduler_;  // last: started after everything above exists
};

}  // namespace lisa
