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
    std::unique_lock<std::mutex> lock(queue_->mutex_);
    while (queue_->serving_ != ticket_ || queue_->running_) {
        if (queue_->closed_) {
            return false;
        }
        lock.unlock();
        const bool stop = give_up && give_up();
        lock.lock();
        if (stop) {
            return false;
        }
        queue_->changed_.wait_for(lock, kPollInterval);
    }
    if (queue_->closed_) {
        return false;
    }
    queue_->running_ = true;
    --queue_->waiting_;
    state_ = State::running;
    const Clock::time_point now = Clock::now();
    waited_ = now - started_;
    started_ = now;
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
            queue_->running_ = false;
            queue_->advance_locked();
        } else {
            --queue_->waiting_;
            if (ticket_ == queue_->serving_) {
                queue_->advance_locked();  // its turn had come: pass it on
            } else {
                queue_->abandoned_.insert(ticket_);
            }
        }
    }
    state_ = State::left;
    queue_->changed_.notify_all();
}

void AdmissionQueue::advance_locked() {
    ++serving_;
    while (abandoned_.erase(serving_) > 0) {
        ++serving_;
    }
}

std::optional<AdmissionQueue::Slot> AdmissionQueue::try_join() {
    const std::lock_guard<std::mutex> lock(mutex_);
    // One request runs; up to max_waiting_ wait behind it.
    const std::size_t in_line = waiting_ + (running_ ? 1 : 0);
    if (closed_ || in_line > max_waiting_) {
        return std::nullopt;
    }
    ++waiting_;
    return Slot(this, next_ticket_++, Clock::now());
}

int AdmissionQueue::retry_after_seconds() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const double estimate = average_service_seconds_ * static_cast<double>(waiting_ + 1);
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
    return waiting_;
}

bool AdmissionQueue::busy() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

bool AdmissionQueue::wait_idle(Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_until(lock, deadline, [this] { return !running_ && waiting_ == 0; });
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
