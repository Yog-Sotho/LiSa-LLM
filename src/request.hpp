#pragma once

#include "utils.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lisa {

// A client error in a request body. `param` names the offending field (OpenAI's
// error.param); the message is safe to return to the client.
class RequestError : public std::invalid_argument {
public:
    RequestError(const std::string& message, std::string param)
        : std::invalid_argument(message), param_(std::move(param)) {}

    [[nodiscard]] const std::string& param() const noexcept { return param_; }

private:
    std::string param_;
};

inline constexpr std::uint32_t kRandomSeed = 0xFFFFFFFFU;  // == LLAMA_DEFAULT_SEED
inline constexpr std::size_t kMaxStopSequences = 4;

struct SamplingParams {
    float temperature = 0.8f;  // 0 = greedy
    float top_p = 1.0f;
    int top_k = 0;  // 0 = disabled
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repeat_penalty = 1.0f;  // 1 = disabled
    std::uint32_t seed = kRandomSeed;
};

struct GenerationParams {
    int max_tokens = 0;
    bool max_tokens_explicit = false;  // false: may be clamped to the remaining context
    std::vector<std::string> stop;
    SamplingParams sampling;
    bool stream = false;
    bool include_usage = false;  // stream_options.include_usage
};

struct ChatMessage {
    std::string role;  // system | user | assistant
    std::string content;
};

struct CompletionRequest {
    std::string prompt;
    GenerationParams params;
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    GenerationParams params;
};

// Parse and validate POST /v1/completions and /v1/chat/completions bodies (OpenAI
// format), filling defaults from `cfg`. Throw RequestError on invalid input.
[[nodiscard]] CompletionRequest parse_completion_request(std::string_view body, const Config& cfg);
[[nodiscard]] ChatRequest parse_chat_request(std::string_view body, const Config& cfg);

}  // namespace lisa
