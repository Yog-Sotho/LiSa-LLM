#pragma once

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

using Token = std::int32_t;  // == llama_token

struct ModelInfo {
    std::string description;
    std::int32_t n_layer = 0;
    std::int32_t n_ctx_train = 0;
    std::uint32_t n_ctx = 0;  // context window actually allocated
    std::uint64_t n_params = 0;
    bool gpu_offload = false;
};

enum class StopReason : std::uint8_t { eos, length, cancelled };

struct GenerationResult {
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    StopReason reason = StopReason::length;
};

// Single-sequence text generation on top of libllama (llama.cpp). Requests are
// serialised; continuous batching is planned (roadmap P4).
class InferenceEngine {
public:
    // Receives UTF-8-complete text; return false to stop generating (e.g. client gone).
    using PieceCallback = std::function<bool(std::string_view piece)>;

    // Loads the GGUF model in cfg.model_path. Throws std::runtime_error on failure.
    explicit InferenceEngine(const Config& cfg);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = delete;
    InferenceEngine& operator=(InferenceEngine&&) = delete;

    // Tokenises a user prompt (BOS added as the model requires; special-token text is
    // treated as plain text). Thread-safe. Throws std::invalid_argument.
    [[nodiscard]] std::vector<Token> tokenize(std::string_view text) const;

    // Generates a completion for `prompt_tokens`. Throws std::invalid_argument when the
    // request does not fit the context window, std::runtime_error on engine failure.
    GenerationResult generate(const std::vector<Token>& prompt_tokens,
                              const CompletionRequest& request, const PieceCallback& on_piece);

    [[nodiscard]] const ModelInfo& info() const noexcept { return info_; }

private:
    struct ModelDeleter {
        void operator()(llama_model* model) const noexcept;
    };
    struct ContextDeleter {
        void operator()(llama_context* ctx) const noexcept;
    };

    [[nodiscard]] std::string token_to_piece(Token token) const;

    std::unique_ptr<llama_model, ModelDeleter> model_;
    std::unique_ptr<llama_context, ContextDeleter> ctx_;
    const llama_vocab* vocab_ = nullptr;
    ModelInfo info_;
    std::mutex mutex_;  // guards ctx_ (one generation at a time)
};

}  // namespace lisa
