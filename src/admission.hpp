#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace lisa {

// First-come, first-served access to the model, with a bounded line. Up to `max_running`
// requests use the model at once (the engine's batch slots); up to `max_waiting` more
// wait in arrival order, and further ones are refused (HTTP 429) instead of piling up
// threads and connections.
class AdmissionQueue {
public:
    using Clock = std::chrono::steady_clock;

    AdmissionQueue(std::size_t max_running, std::size_t max_waiting)
        : max_running_(std::max<std::size_t>(1, max_running)), max_waiting_(max_waiting) {}

    // A place in line. Destroying it leaves the line (or hands the model to the next).
    class Slot {
    public:
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
        Slot(Slot&& other) noexcept
            : queue_(std::exchange(other.queue_, nullptr)),
              ticket_(other.ticket_),
              state_(std::exchange(other.state_, State::left)),
              started_(other.started_),
              waited_(other.waited_) {}
        Slot& operator=(Slot&&) = delete;
        ~Slot() { leave(); }

        // Blocks until it is this slot's turn. Returns false if `give_up` returned true
        // first (checked every 100 ms) or the queue was closed.
        [[nodiscard]] bool wait(const std::function<bool()>& give_up);

        // Time spent waiting in line (valid after wait() returned true).
        [[nodiscard]] Clock::duration waited() const noexcept { return waited_; }

    private:
        friend class AdmissionQueue;
        enum class State : std::uint8_t { waiting, running, left };

        Slot(AdmissionQueue* queue, std::uint64_t ticket, Clock::time_point joined)
            : queue_(queue), ticket_(ticket), started_(joined) {}
        void leave() noexcept;

        AdmissionQueue* queue_;
        std::uint64_t ticket_;
        State state_ = State::waiting;
        Clock::time_point started_;
        Clock::duration waited_{};
    };

    // Joins the line, or returns std::nullopt when it is full or the queue is closed.
    [[nodiscard]] std::optional<Slot> try_join();

    // Seconds a refused client should wait before retrying, from recent service times.
    [[nodiscard]] int retry_after_seconds() const;

    // Refuses new requests and wakes the waiting ones (shutdown).
    void close();

    [[nodiscard]] std::size_t waiting() const;
    [[nodiscard]] std::size_t running() const;

    // Blocks until nothing is running or waiting, or `deadline` passes. Returns true if idle.
    [[nodiscard]] bool wait_idle(Clock::time_point deadline);

private:
    const std::size_t max_running_;
    const std::size_t max_waiting_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::uint64_t next_ticket_ = 0;
    std::set<std::uint64_t> waiting_;  // tickets in line, oldest first
    std::size_t running_ = 0;
    bool closed_ = false;
    double average_service_seconds_ = 1.0;  // exponential moving average
};

// Token-bucket rate limiting per client (API key id, or address without auth).
class RateLimiter {
public:
    using Clock = std::chrono::steady_clock;

    // `per_minute` == 0 disables limiting.
    RateLimiter(int per_minute, int burst);

    // 0 if the request may proceed, else the seconds until the client may retry.
    [[nodiscard]] int acquire(const std::string& client, Clock::time_point now = Clock::now());

    [[nodiscard]] bool enabled() const noexcept { return per_second_ > 0; }

private:
    struct Bucket {
        double tokens = 0;
        Clock::time_point updated;
    };

    void prune_locked(Clock::time_point now);

    double per_second_;
    double burst_;
    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};

}  // namespace lisa
