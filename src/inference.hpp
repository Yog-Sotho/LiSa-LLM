#pragma once

#include "engine.hpp"
#include "request.hpp"
#include "utils.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace lisa {

// Single-sequence text generation on top of libllama (llama.cpp). Requests are
// serialised; continuous batching is planned (roadmap P4).
class InferenceEngine final : public Engine {
public:
    // Loads the GGUF model in cfg.model_path and selects the chat template
    // (cfg.chat_template, else the model's own, else ChatML). Throws std::runtime_error.
    explicit InferenceEngine(const Config& cfg);
    ~InferenceEngine() override;

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

    // Generates a completion for `prompt_tokens`. Throws RequestError when the request
    // does not fit the context window, std::runtime_error on engine failure.
    GenerationResult generate(const std::vector<Token>& prompt_tokens,
                              const GenerationParams& params,
                              const PieceCallback& on_piece) override;

    [[nodiscard]] ModelInfo info() const override { return info_; }

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

    [[nodiscard]] std::vector<Token> tokenize_text(std::string_view text, bool add_special,
                                                   bool parse_special) const;
    [[nodiscard]] std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;
    [[nodiscard]] std::string token_to_piece(Token token) const;
    void select_chat_template(const std::string& configured);

    std::unique_ptr<llama_model, ModelDeleter> model_;
    std::unique_ptr<llama_context, ContextDeleter> ctx_;
    const llama_vocab* vocab_ = nullptr;
    std::string chat_template_;  // template text or built-in name for llama_chat_apply_template
    ModelInfo info_;
    std::mutex mutex_;  // guards ctx_ (one generation at a time)
};

}  // namespace lisa
