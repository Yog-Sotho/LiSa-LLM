#pragma once

#include "request.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace lisa {

using Token = std::int32_t;  // == llama_token

struct ModelInfo {
    std::string description;
    std::string chat_template;  // how chat prompts are formatted (for logs)
    std::int32_t n_layer = 0;
    std::int32_t n_ctx_train = 0;
    std::uint32_t n_ctx = 0;  // context window requests are limited to
    std::uint64_t n_params = 0;
    bool gpu_offload = false;
};

// Health of the engine, readable without waiting for a running generation.
struct EngineStatus {
    bool ready = true;                // can serve requests (now, or after a restart)
    std::uint64_t worker_deaths = 0;  // sandboxed worker crashes or seccomp kills
    std::uint64_t worker_restarts = 0;
};

enum class StopReason : std::uint8_t { eos, stop_sequence, length, cancelled };

struct GenerationResult {
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    StopReason reason = StopReason::length;
};

// What the HTTP layer needs from a model. Implemented in-process by InferenceEngine and
// across the sandbox boundary by SandboxedEngine.
class Engine {
public:
    // Receives UTF-8-complete text; return false to stop generating (e.g. client gone).
    using PieceCallback = std::function<bool(std::string_view piece)>;

    virtual ~Engine() = default;

    // Tokenises a raw completion prompt; special-token text stays plain text.
    // Throws RequestError.
    [[nodiscard]] virtual std::vector<Token> tokenize(std::string_view text) = 0;

    // Formats `messages` with the chat template and tokenises the result; content that
    // spells a special token is rejected. Throws RequestError.
    [[nodiscard]] virtual std::vector<Token> tokenize_chat(
        const std::vector<ChatMessage>& messages) = 0;

    // Generates a completion. Throws RequestError when the request does not fit the
    // context window, std::runtime_error on engine failure.
    virtual GenerationResult generate(const std::vector<Token>& prompt_tokens,
                                      const GenerationParams& params,
                                      const PieceCallback& on_piece) = 0;

    [[nodiscard]] virtual ModelInfo info() const = 0;

    [[nodiscard]] virtual EngineStatus status() const noexcept { return {}; }
};

}  // namespace lisa
