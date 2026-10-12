#include "openai.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <utility>

namespace lisa::openai {

namespace {

constexpr std::size_t kIdHexDigits = 24;

json choice_base(const json& finish) {
    return json{{"index", 0}, {"finish_reason", finish}};
}

}  // namespace

std::string random_hex(std::size_t digits) {
    static constexpr std::array<char, 16> kHex = {'0', '1', '2', '3', '4', '5', '6', '7',
                                                  '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> dist(0, kHex.size() - 1);
    std::string out(digits, '0');
    std::generate(out.begin(), out.end(), [&] { return kHex[dist(rng)]; });
    return out;
}

ResponseMeta make_meta(std::string_view id_prefix, std::string model) {
    ResponseMeta meta;
    meta.id = std::string(id_prefix) + random_hex(kIdHexDigits);
    meta.created = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    meta.model = std::move(model);
    return meta;
}

json finish_reason(StopReason reason) {
    switch (reason) {
        case StopReason::eos:
        case StopReason::stop_sequence:
            return "stop";
        case StopReason::length:
            return "length";
        case StopReason::cancelled:
            break;
    }
    return nullptr;
}

json usage(const GenerationResult& result) {
    return json{{"prompt_tokens", result.prompt_tokens},
                {"completion_tokens", result.completion_tokens},
                {"total_tokens", result.prompt_tokens + result.completion_tokens}};
}

json completion(const ResponseMeta& meta, const std::string& text, const GenerationResult& result) {
    json choice = choice_base(finish_reason(result.reason));
    choice["text"] = text;
    choice["logprobs"] = nullptr;
    return json{
        {"id", meta.id},       {"object", "text_completion"},      {"created", meta.created},
        {"model", meta.model}, {"choices", json::array({choice})}, {"usage", usage(result)}};
}

json completion_chunk(const ResponseMeta& meta, const std::string& text, const json& finish) {
    json choice = choice_base(finish);
    choice["text"] = text;
    choice["logprobs"] = nullptr;
    return json{{"id", meta.id},
                {"object", "text_completion"},
                {"created", meta.created},
                {"model", meta.model},
                {"choices", json::array({choice})}};
}

json chat_completion(const ResponseMeta& meta, const std::string& content,
                     const GenerationResult& result) {
    json choice = choice_base(finish_reason(result.reason));
    choice["message"] = {{"role", "assistant"}, {"content", content}};
    choice["logprobs"] = nullptr;
    return json{
        {"id", meta.id},       {"object", "chat.completion"},      {"created", meta.created},
        {"model", meta.model}, {"choices", json::array({choice})}, {"usage", usage(result)}};
}

json chat_chunk(const ResponseMeta& meta, const json& delta, const json& finish) {
    json choice = choice_base(finish);
    choice["delta"] = delta;
    choice["logprobs"] = nullptr;
    return json{{"id", meta.id},
                {"object", "chat.completion.chunk"},
                {"created", meta.created},
                {"model", meta.model},
                {"choices", json::array({choice})}};
}

json usage_chunk(const ResponseMeta& meta, std::string_view object,
                 const GenerationResult& result) {
    return json{{"id", meta.id},       {"object", object},         {"created", meta.created},
                {"model", meta.model}, {"choices", json::array()}, {"usage", usage(result)}};
}

json models(const std::string& model_id, std::int64_t created) {
    return json{
        {"object", "list"},
        {"data",
         json::array({json{
             {"id", model_id}, {"object", "model"}, {"created", created}, {"owned_by", "lisa"}}})}};
}

json error(std::string_view message, std::string_view type, std::string_view param,
           std::string_view code) {
    return json{{"error",
                 {{"message", message},
                  {"type", type},
                  {"param", param.empty() ? json(nullptr) : json(param)},
                  {"code", code.empty() ? json(nullptr) : json(code)}}}};
}

}  // namespace lisa::openai
