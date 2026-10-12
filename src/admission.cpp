#include "admission.hpp"

#include <algorithm>
#include <cmath>

namespace lisa {

namespace {

constexpr auto kPollInterval = std::chrono::milliseconds(100);
constexpr double kServiceTimeWeight = 0.2;  // weight of the newest sample in the average
constexpr int kMaxRetryAfterSeconds = 60;
constexpr std::size_t kMaxTrackedClients = 10'000;
constexpr double kSecondsPerMinute = 60.0;

}  // namespace

bool AdmissionQueue::Slot::wait(const std::function<bool()>& give_up) {
    if (queue_ == nullptr || state_ != State::waiting) {
        return state_ == State::running;
    }
    AdmissionQueue& q = *queue_;
    std::unique_lock<std::mutex> lock(q.mutex_);
    // Runs when it is the oldest ticket in line and a slot is free.
    while (*q.waiting_.begin() != ticket_ || q.running_ >= q.max_running_) {
        if (q.closed_) {
            return false;
        }
        lock.unlock();
        const bool stop = give_up && give_up();
        lock.lock();
        if (stop) {
            return false;
        }
        q.changed_.wait_for(lock, kPollInterval);
    }
    if (q.closed_) {
        return false;
    }
    q.waiting_.erase(ticket_);
    ++q.running_;
    state_ = State::running;
    const Clock::time_point now = Clock::now();
    waited_ = now - started_;
    started_ = now;
    lock.unlock();
    q.changed_.notify_all();  // the next ticket may run too if another slot is free
    return true;
}

void AdmissionQueue::Slot::leave() noexcept {
    if (queue_ == nullptr || state_ == State::left) {
        return;
    }
    {
        const std::lock_guard<std::mutex> lock(queue_->mutex_);
        if (state_ == State::running) {
            const double seconds = std::chrono::duration<double>(Clock::now() - started_).count();
            queue_->average_service_seconds_ +=
                kServiceTimeWeight * (seconds - queue_->average_service_seconds_);
            --queue_->running_;
        } else {
            queue_->waiting_.erase(ticket_);
        }
    }
    state_ = State::left;
    queue_->changed_.notify_all();
}

std::optional<AdmissionQueue::Slot> AdmissionQueue::try_join() {
    const std::lock_guard<std::mutex> lock(mutex_);
    // Requests that can start at once don't count against the waiting limit.
    const std::size_t free_slots = running_ < max_running_ ? max_running_ - running_ : 0;
    const std::size_t in_line = waiting_.size() + 1;
    const std::size_t would_wait = in_line > free_slots ? in_line - free_slots : 0;
    if (closed_ || would_wait > max_waiting_) {
        return std::nullopt;
    }
    const std::uint64_t ticket = next_ticket_++;
    waiting_.insert(ticket);
    return Slot(this, ticket, Clock::now());
}

int AdmissionQueue::retry_after_seconds() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    // Requests ahead, divided among the slots, times the average service time.
    const double rounds =
        static_cast<double>(waiting_.size() + 1) / static_cast<double>(max_running_);
    const double estimate = average_service_seconds_ * std::max(1.0, rounds);
    return std::clamp(static_cast<int>(std::ceil(estimate)), 1, kMaxRetryAfterSeconds);
}

void AdmissionQueue::close() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    changed_.notify_all();
}

std::size_t AdmissionQueue::waiting() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return waiting_.size();
}

std::size_t AdmissionQueue::running() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

bool AdmissionQueue::wait_idle(Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_until(lock, deadline,
                               [this] { return running_ == 0 && waiting_.empty(); });
}

RateLimiter::RateLimiter(int per_minute, int burst)
    : per_second_(per_minute / kSecondsPerMinute), burst_(std::max(1, burst)) {
}

int RateLimiter::acquire(const std::string& client, Clock::time_point now) {
    if (!enabled()) {
        return 0;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (buckets_.size() >= kMaxTrackedClients && !buckets_.contains(client)) {
        prune_locked(now);
    }
    auto [it, inserted] = buckets_.try_emplace(client, Bucket{burst_, now});
    Bucket& bucket = it->second;
    if (!inserted) {
        const double elapsed = std::chrono::duration<double>(now - bucket.updated).count();
        bucket.tokens = std::min(burst_, bucket.tokens + std::max(0.0, elapsed) * per_second_);
        bucket.updated = now;
    }
    if (bucket.tokens >= 1.0) {
        bucket.tokens -= 1.0;
        return 0;
    }
    return std::max(1, static_cast<int>(std::ceil((1.0 - bucket.tokens) / per_second_)));
}

void RateLimiter::prune_locked(Clock::time_point now) {
    // Buckets that have refilled completely carry no state worth keeping.
    std::erase_if(buckets_, [&](const auto& entry) {
        const double elapsed = std::chrono::duration<double>(now - entry.second.updated).count();
        return entry.second.tokens + elapsed * per_second_ >= burst_;
    });
    if (buckets_.size() >= kMaxTrackedClients) {
        buckets_.clear();  // a flood of distinct clients: forget all rather than grow
    }
}

}  // namespace lisa
