#pragma once

#include "utils.hpp"

#include <string>
#include <string_view>

namespace lisa {

struct CompletionRequest {
    std::string prompt;
    int max_new_tokens = 0;
    float temperature = 0.0f;  // 0 = greedy
    float top_p = 1.0f;
};

// Parses and validates the body of POST /v1/completions, filling defaults from `cfg`.
// Throws std::invalid_argument with a message that is safe to return to the client.
[[nodiscard]] CompletionRequest parse_completion_request(std::string_view body, const Config& cfg);

}  // namespace lisa
