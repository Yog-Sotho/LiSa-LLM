#include "metrics.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <lisa/version.hpp>

namespace lisa {

namespace {

// Seconds: from fast API calls to long generations.
const std::vector<double> kLatencyBuckets = {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                                             1,     2.5,  5,     10,   30,  60,   120};
const std::vector<double> kRateBuckets = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

std::string number(double value) {
    if (std::isinf(value)) {
        return value > 0 ? "+Inf" : "-Inf";
    }
    std::array<char, 32> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return error == std::errc() ? std::string(buffer.data(), end) : std::string("NaN");
}

void header(std::string& out, std::string_view name, std::string_view type, std::string_view help) {
    out.append("# HELP ").append(name).append(" ").append(help).append("\n");
    out.append("# TYPE ").append(name).append(" ").append(type).append("\n");
}

void sample(std::string& out, std::string_view name, std::string_view labels, double value) {
    out.append(name);
    if (!labels.empty()) {
        out.append("{").append(labels).append("}");
    }
    out.append(" ").append(number(value)).append("\n");
}

std::string label(std::string_view key, std::string_view value) {
    // Values come from a fixed set (routes, reasons), so no escaping is needed.
    return std::string(key) + "=\"" + std::string(value) + "\"";
}

}  // namespace

Histogram::Histogram(std::vector<double> upper_bounds)
    : bounds_(std::move(upper_bounds)), counts_(bounds_.size() + 1, 0) {
}

void Histogram::observe(double value) {
    std::size_t i = 0;
    while (i < bounds_.size() && value > bounds_[i]) {
        ++i;
    }
    ++counts_[i];
    sum_ += value;
    ++count_;
}

void Histogram::render(std::string& out, std::string_view name, std::string_view labels) const {
    const std::string bucket = std::string(name) + "_bucket";
    const std::string prefix = labels.empty() ? std::string() : std::string(labels) + ",";
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i <= bounds_.size(); ++i) {
        cumulative += counts_[i];
        const double bound =
            i < bounds_.size() ? bounds_[i] : std::numeric_limits<double>::infinity();
        sample(out, bucket, prefix + label("le", number(bound)), static_cast<double>(cumulative));
    }
    sample(out, std::string(name) + "_sum", labels, sum_);
    sample(out, std::string(name) + "_count", labels, static_cast<double>(count_));
}

Metrics::Metrics()
    : queue_seconds_(kLatencyBuckets),
      first_token_seconds_(kLatencyBuckets),
      tokens_per_second_(kRateBuckets) {
}

void Metrics::request_finished(std::string_view route, int status, double seconds) {
    const std::lock_guard<std::mutex> lock(mutex_);
    ++requests_[{std::string(route), status}];
    request_seconds_.try_emplace(std::string(route), kLatencyBuckets)
        .first->second.observe(seconds);
}

void Metrics::rejected(std::string_view reason) {
    const std::lock_guard<std::mutex> lock(mutex_);
    ++rejected_[std::string(reason)];
}

void Metrics::generation_finished(std::size_t prompt_tokens, std::size_t completion_tokens,
                                  double queue_seconds, double first_token_seconds,
                                  double generation_seconds) {
    const std::lock_guard<std::mutex> lock(mutex_);
    prompt_tokens_ += prompt_tokens;
    completion_tokens_ += completion_tokens;
    ++generations_;
    queue_seconds_.observe(queue_seconds);
    if (completion_tokens > 0) {
        first_token_seconds_.observe(first_token_seconds);
        if (generation_seconds > 0) {
            tokens_per_second_.observe(static_cast<double>(completion_tokens) / generation_seconds);
        }
    }
}

std::string Metrics::render(const MetricsGauges& gauges) const {
    std::string out;
    header(out, "lisa_build_info", "gauge", "Version of the running server.");
    sample(out, "lisa_build_info", label("version", kVersion), 1);

    header(out, "lisa_ready", "gauge", "1 when the server accepts inference requests.");
    sample(out, "lisa_ready", {}, gauges.engine.ready && !gauges.draining ? 1 : 0);
    header(out, "lisa_requests_running", "gauge", "Requests using the model right now.");
    sample(out, "lisa_requests_running", {}, gauges.busy ? 1 : 0);
    header(out, "lisa_requests_queued", "gauge", "Requests waiting for the model.");
    sample(out, "lisa_requests_queued", {}, static_cast<double>(gauges.queued));
    header(out, "lisa_worker_deaths_total", "counter",
           "Sandboxed worker crashes, including seccomp kills.");
    sample(out, "lisa_worker_deaths_total", {}, static_cast<double>(gauges.engine.worker_deaths));
    header(out, "lisa_worker_restarts_total", "counter", "Sandboxed worker restarts.");
    sample(out, "lisa_worker_restarts_total", {},
           static_cast<double>(gauges.engine.worker_restarts));

    const std::lock_guard<std::mutex> lock(mutex_);
    header(out, "lisa_http_requests_total", "counter", "HTTP requests by route and status.");
    for (const auto& [key, count] : requests_) {
        sample(out, "lisa_http_requests_total",
               label("route", key.first) + "," + label("code", std::to_string(key.second)),
               static_cast<double>(count));
    }
    header(out, "lisa_http_request_duration_seconds", "histogram",
           "Time from request to the end of the response, by route.");
    for (const auto& [route, histogram] : request_seconds_) {
        histogram.render(out, "lisa_http_request_duration_seconds", label("route", route));
    }
    header(out, "lisa_requests_rejected_total", "counter",
           "Requests refused before reaching the model, by reason.");
    for (const auto& [reason, count] : rejected_) {
        sample(out, "lisa_requests_rejected_total", label("reason", reason),
               static_cast<double>(count));
    }
    header(out, "lisa_generations_total", "counter", "Completed generations.");
    sample(out, "lisa_generations_total", {}, static_cast<double>(generations_));
    header(out, "lisa_prompt_tokens_total", "counter", "Prompt tokens processed.");
    sample(out, "lisa_prompt_tokens_total", {}, static_cast<double>(prompt_tokens_));
    header(out, "lisa_completion_tokens_total", "counter", "Tokens generated.");
    sample(out, "lisa_completion_tokens_total", {}, static_cast<double>(completion_tokens_));
    header(out, "lisa_queue_wait_seconds", "histogram", "Time requests waited for the model.");
    queue_seconds_.render(out, "lisa_queue_wait_seconds", {});
    header(out, "lisa_time_to_first_token_seconds", "histogram",
           "Time from getting the model to the first generated text.");
    first_token_seconds_.render(out, "lisa_time_to_first_token_seconds", {});
    header(out, "lisa_generation_tokens_per_second", "histogram",
           "Generation speed per request (completion tokens / generation time).");
    tokens_per_second_.render(out, "lisa_generation_tokens_per_second", {});
    return out;
}

}  // namespace lisa
