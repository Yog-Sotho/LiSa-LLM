#pragma once

#include "engine.hpp"
#include "utils.hpp"

#include <cstdint>
#include <string>
#include <string_view>

// JSON bodies in the OpenAI API format (completions, chat completions, models, errors).
namespace lisa::openai {

// Identifies one response: all chunks of a stream share it.
struct ResponseMeta {
    std::string id;            // "cmpl-…" / "chatcmpl-…"
    std::int64_t created = 0;  // Unix seconds
    std::string model;
};

[[nodiscard]] ResponseMeta make_meta(std::string_view id_prefix, std::string model);

// "stop" | "length"; cancelled generations have no finish reason.
[[nodiscard]] json finish_reason(StopReason reason);
[[nodiscard]] json usage(const GenerationResult& result);

[[nodiscard]] json completion(const ResponseMeta& meta, const std::string& text,
                              const GenerationResult& result);
[[nodiscard]] json completion_chunk(const ResponseMeta& meta, const std::string& text,
                                    const json& finish = nullptr);

[[nodiscard]] json chat_completion(const ResponseMeta& meta, const std::string& content,
                                   const GenerationResult& result);
[[nodiscard]] json chat_chunk(const ResponseMeta& meta, const json& delta,
                              const json& finish = nullptr);

// Final stream chunk for stream_options.include_usage (empty choices).
[[nodiscard]] json usage_chunk(const ResponseMeta& meta, std::string_view object,
                               const GenerationResult& result);

[[nodiscard]] json models(const std::string& model_id, std::int64_t created);

[[nodiscard]] json error(std::string_view message, std::string_view type,
                         std::string_view param = {}, std::string_view code = {});

}  // namespace lisa::openai
