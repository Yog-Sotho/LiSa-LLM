#include "request.hpp"

#include <cmath>
#include <stdexcept>

namespace lisa {

namespace {

constexpr double kMaxTemperature = 2.0;

double read_number(const json& payload, const char* key, double fallback) {
    const auto it = payload.find(key);
    if (it == payload.end()) {
        return fallback;
    }
    if (!it->is_number()) {
        throw std::invalid_argument(std::string(key) + " must be a number");
    }
    const double value = it->get<double>();
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(key) + " must be finite");
    }
    return value;
}

}  // namespace

CompletionRequest parse_completion_request(std::string_view body, const Config& cfg) {
    const json payload = json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (payload.is_discarded()) {
        throw std::invalid_argument("invalid JSON");
    }
    if (!payload.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }

    CompletionRequest req;

    const auto prompt = payload.find("prompt");
    if (prompt == payload.end() || !prompt->is_string() ||
        prompt->get_ref<const std::string&>().empty()) {
        throw std::invalid_argument("prompt must be a non-empty string");
    }
    req.prompt = prompt->get<std::string>();

    req.max_new_tokens = cfg.max_new_tokens_default;
    if (const auto it = payload.find("max_new_tokens"); it != payload.end()) {
        if (!it->is_number_integer()) {
            throw std::invalid_argument("max_new_tokens must be an integer");
        }
        const auto value = it->get<long long>();
        if (value < 1 || value > static_cast<long long>(cfg.max_context)) {
            throw std::invalid_argument("max_new_tokens must be between 1 and " +
                                        std::to_string(cfg.max_context));
        }
        req.max_new_tokens = static_cast<int>(value);
    }

    const double temperature =
        read_number(payload, "temperature", static_cast<double>(cfg.temperature_default));
    if (temperature < 0.0 || temperature > kMaxTemperature) {
        throw std::invalid_argument("temperature must be between 0 and 2");
    }
    req.temperature = static_cast<float>(temperature);

    const double top_p = read_number(payload, "top_p", static_cast<double>(cfg.top_p_default));
    if (top_p <= 0.0 || top_p > 1.0) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    req.top_p = static_cast<float>(top_p);

    return req;
}

}  // namespace lisa
