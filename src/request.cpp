#include "request.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace lisa {

namespace {

constexpr double kMaxTemperature = 2.0;
constexpr double kMaxOpenAiPenalty = 2.0;  // presence/frequency: [-2, 2]
constexpr double kMaxRepeatPenalty = 2.0;
constexpr long long kMaxTopK = 1000;
constexpr std::size_t kMaxStopLength = 256;

std::string range_message(const char* key, const std::string& range) {
    return std::string(key) + " must be " + range;
}

json parse_object(std::string_view body) {
    json payload = json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (payload.is_discarded()) {
        throw RequestError("invalid JSON", "");
    }
    if (!payload.is_object()) {
        throw RequestError("request body must be a JSON object", "");
    }
    return payload;
}

// Returns the value of `key`, or nullptr when absent or JSON null (= use default).
const json* find_value(const json& payload, const char* key) {
    const auto it = payload.find(key);
    if (it == payload.end() || it->is_null()) {
        return nullptr;
    }
    return &*it;
}

std::optional<double> read_number(const json& payload, const char* key, double lo, double hi,
                                  bool lo_inclusive = true) {
    const json* value = find_value(payload, key);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (!value->is_number()) {
        throw RequestError(std::string(key) + " must be a number", key);
    }
    const double number = value->get<double>();
    const bool above_lo = lo_inclusive ? number >= lo : number > lo;
    if (!std::isfinite(number) || !above_lo || number > hi) {
        throw RequestError(range_message(key, (lo_inclusive ? "in [" : "in (") + json(lo).dump() +
                                                  ", " + json(hi).dump() + "]"),
                           key);
    }
    return number;
}

std::optional<long long> read_integer(const json& payload, const char* key, long long lo,
                                      long long hi) {
    const json* value = find_value(payload, key);
    if (value == nullptr) {
        return std::nullopt;
    }
    // Unsigned values above LLONG_MAX are out of range for every field here.
    if (!value->is_number_integer() ||
        (value->is_number_unsigned() &&
         value->get<unsigned long long>() >
             static_cast<unsigned long long>(std::numeric_limits<long long>::max()))) {
        throw RequestError(range_message(key, "an integer between " + std::to_string(lo) + " and " +
                                                  std::to_string(hi)),
                           key);
    }
    const auto number = value->get<long long>();
    if (number < lo || number > hi) {
        throw RequestError(
            range_message(key, "between " + std::to_string(lo) + " and " + std::to_string(hi)),
            key);
    }
    return number;
}

bool read_bool(const json& payload, const char* key, bool fallback) {
    const json* value = find_value(payload, key);
    if (value == nullptr) {
        return fallback;
    }
    if (!value->is_boolean()) {
        throw RequestError(std::string(key) + " must be a boolean", key);
    }
    return value->get<bool>();
}

std::vector<std::string> read_stop(const json& payload) {
    const json* value = find_value(payload, "stop");
    if (value == nullptr) {
        return {};
    }
    std::vector<std::string> stop;
    const auto add = [&stop](const json& item) {
        if (!item.is_string() || item.get_ref<const std::string&>().empty() ||
            item.get_ref<const std::string&>().size() > kMaxStopLength) {
            throw RequestError("stop entries must be non-empty strings of at most " +
                                   std::to_string(kMaxStopLength) + " bytes",
                               "stop");
        }
        stop.push_back(item.get<std::string>());
    };
    if (value->is_array()) {
        if (value->size() > kMaxStopSequences) {
            throw RequestError(
                "stop accepts at most " + std::to_string(kMaxStopSequences) + " sequences", "stop");
        }
        for (const json& item : *value) {
            add(item);
        }
    } else {
        add(*value);
    }
    return stop;
}

// Fields with OpenAI semantics this server does not implement: reject rather than ignore,
// so clients don't silently get different behaviour.
void reject_unsupported(const json& payload) {
    for (const char* key : {"n", "best_of"}) {
        const json* value = find_value(payload, key);
        if (value != nullptr && *value != 1) {
            throw RequestError(std::string("only ") + key + " = 1 is supported", key);
        }
    }
    for (const char* key : {"logprobs", "echo"}) {
        if (read_bool(payload, key, false)) {
            throw RequestError(std::string(key) + " is not supported", key);
        }
    }
    for (const char* key : {"top_logprobs", "suffix", "functions"}) {
        if (find_value(payload, key) != nullptr) {
            throw RequestError(std::string(key) + " is not supported", key);
        }
    }
    for (const char* key : {"tools", "logit_bias"}) {
        const json* value = find_value(payload, key);
        if (value != nullptr && !value->empty()) {
            throw RequestError(std::string(key) + " is not supported", key);
        }
    }
    if (const json* format = find_value(payload, "response_format")) {
        if (!format->is_object() || format->value("type", "") != "text") {
            throw RequestError(R"(only response_format {"type": "text"} is supported)",
                               "response_format");
        }
    }
}

GenerationParams parse_generation_params(const json& payload, const Config& cfg) {
    reject_unsupported(payload);

    GenerationParams params;
    params.max_tokens = cfg.max_new_tokens_default;
    const auto max_context = static_cast<long long>(cfg.max_context);
    for (const char* key : {"max_completion_tokens", "max_tokens", "max_new_tokens"}) {
        if (const auto value = read_integer(payload, key, 1, max_context)) {
            params.max_tokens = static_cast<int>(*value);
            params.max_tokens_explicit = true;
            break;
        }
    }

    SamplingParams& s = params.sampling;
    s.temperature = static_cast<float>(read_number(payload, "temperature", 0.0, kMaxTemperature)
                                           .value_or(cfg.temperature_default));
    s.top_p = static_cast<float>(read_number(payload, "top_p", 0.0, 1.0, /*lo_inclusive=*/false)
                                     .value_or(cfg.top_p_default));
    s.top_k = static_cast<int>(read_integer(payload, "top_k", 0, kMaxTopK).value_or(0));
    s.min_p = static_cast<float>(read_number(payload, "min_p", 0.0, 1.0).value_or(0.0));
    s.presence_penalty = static_cast<float>(
        read_number(payload, "presence_penalty", -kMaxOpenAiPenalty, kMaxOpenAiPenalty)
            .value_or(0.0));
    s.frequency_penalty = static_cast<float>(
        read_number(payload, "frequency_penalty", -kMaxOpenAiPenalty, kMaxOpenAiPenalty)
            .value_or(0.0));
    s.repeat_penalty = static_cast<float>(
        read_number(payload, "repeat_penalty", 0.0, kMaxRepeatPenalty, /*lo_inclusive=*/false)
            .value_or(1.0));
    // kRandomSeed itself is reserved for "pick a random seed".
    if (const auto seed = read_integer(payload, "seed", 0, kRandomSeed - 1LL)) {
        s.seed = static_cast<std::uint32_t>(*seed);
    }

    params.stop = read_stop(payload);
    params.stream = read_bool(payload, "stream", false);
    if (const json* options = find_value(payload, "stream_options")) {
        if (!options->is_object()) {
            throw RequestError("stream_options must be an object", "stream_options");
        }
        params.include_usage = read_bool(*options, "include_usage", false);
    }
    return params;
}

std::string read_text_content(const json& content, const std::string& param) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string text;
        for (const json& part : content) {
            if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text") ||
                !part["text"].is_string()) {
                throw RequestError("only text content parts are supported", param);
            }
            text += part["text"].get_ref<const std::string&>();
        }
        return text;
    }
    throw RequestError("content must be a string or an array of text parts", param);
}

}  // namespace

CompletionRequest parse_completion_request(std::string_view body, const Config& cfg) {
    const json payload = parse_object(body);

    CompletionRequest request;
    const json* prompt = find_value(payload, "prompt");
    if (prompt != nullptr && prompt->is_array() && prompt->size() == 1) {
        prompt = &(*prompt)[0];
    }
    if (prompt == nullptr || !prompt->is_string() ||
        prompt->get_ref<const std::string&>().empty()) {
        throw RequestError("prompt must be a non-empty string", "prompt");
    }
    request.prompt = prompt->get<std::string>();
    request.params = parse_generation_params(payload, cfg);
    return request;
}

ChatRequest parse_chat_request(std::string_view body, const Config& cfg) {
    const json payload = parse_object(body);

    const json* messages = find_value(payload, "messages");
    if (messages == nullptr || !messages->is_array() || messages->empty()) {
        throw RequestError("messages must be a non-empty array", "messages");
    }

    ChatRequest request;
    for (std::size_t i = 0; i < messages->size(); ++i) {
        const json& message = (*messages)[i];
        const std::string param = "messages[" + std::to_string(i) + "]";
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string()) {
            throw RequestError(param + " must be an object with a string role", param);
        }
        std::string role = message["role"].get<std::string>();
        if (role == "developer") {
            role = "system";  // OpenAI's newer name for system instructions
        }
        if (role != "system" && role != "user" && role != "assistant") {
            throw RequestError(param + ".role must be system, developer, user or assistant",
                               param + ".role");
        }
        const json* content = find_value(message, "content");
        if (content == nullptr) {
            throw RequestError(param + ".content is required", param + ".content");
        }
        request.messages.push_back(
            {std::move(role), read_text_content(*content, param + ".content")});
    }
    request.params = parse_generation_params(payload, cfg);
    return request;
}

}  // namespace lisa
