#pragma once

#include "engine.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lisa {

// Fixed-bucket histogram in Prometheus' cumulative format.
class Histogram {
public:
    explicit Histogram(std::vector<double> upper_bounds);

    void observe(double value);
    void render(std::string& out, std::string_view name, std::string_view labels) const;

private:
    std::vector<double> bounds_;
    std::vector<std::uint64_t> counts_;  // per bucket, plus +Inf
    double sum_ = 0;
    std::uint64_t count_ = 0;
};

// Values that live elsewhere and are sampled when /metrics is scraped.
struct MetricsGauges {
    std::size_t queued = 0;
    std::size_t running = 0;
    bool draining = false;
    EngineStatus engine;
};

// Server metrics in the Prometheus text exposition format (version 0.0.4).
class Metrics {
public:
    Metrics();

    // `route` must be one of the server's routes (or "other") to bound label cardinality.
    void request_finished(std::string_view route, int status, double seconds);
    void rejected(std::string_view reason);  // queue_full, rate_limited, unauthorized, draining
    void generation_finished(const GenerationResult& result, double queue_seconds,
                             double first_token_seconds, double generation_seconds);

    [[nodiscard]] std::string render(const MetricsGauges& gauges) const;

private:
    mutable std::mutex mutex_;
    std::map<std::pair<std::string, int>, std::uint64_t> requests_;
    std::map<std::string, Histogram> request_seconds_;
    std::map<std::string, std::uint64_t> rejected_;
    std::uint64_t prompt_tokens_ = 0;
    std::uint64_t cached_tokens_ = 0;
    std::uint64_t completion_tokens_ = 0;
    std::uint64_t generations_ = 0;
    Histogram queue_seconds_;
    Histogram first_token_seconds_;
    Histogram tokens_per_second_;
};

}  // namespace lisa
