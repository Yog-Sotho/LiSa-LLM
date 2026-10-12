#include "admission.hpp"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using lisa::AdmissionQueue;

// The optional's value; throws (failing the test) when it is empty.
template <typename T>
T& value_of(std::optional<T>& optional) {
    if (!optional.has_value()) {
        throw std::logic_error("expected a value");
    }
    return *optional;
}

bool never() {
    return false;
}

}  // namespace

TEST_CASE("admission: one runs, max_waiting wait, the rest are refused", "[admission]") {
    AdmissionQueue queue(1);
    std::optional<AdmissionQueue::Slot> a = queue.try_join();
    REQUIRE(a);
    REQUIRE(value_of(a).wait(never));
    CHECK(queue.busy());

    std::optional<AdmissionQueue::Slot> b = queue.try_join();
    REQUIRE(b);
    CHECK(queue.waiting() == 1);
    CHECK_FALSE(queue.try_join());  // one running + one waiting = full
    CHECK(queue.retry_after_seconds() >= 1);

    a.reset();
    REQUIRE(value_of(b).wait(never));
    CHECK(queue.waiting() == 0);
    CHECK(queue.try_join());
}

TEST_CASE("admission: first come, first served", "[admission]") {
    AdmissionQueue queue(8);
    std::optional<AdmissionQueue::Slot> first = queue.try_join();
    REQUIRE(value_of(first).wait(never));

    std::mutex mutex;
    std::vector<int> order;
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        // Join in order on this thread, wait on another.
        std::optional<AdmissionQueue::Slot> joined = queue.try_join();
        auto slot = std::make_shared<AdmissionQueue::Slot>(std::move(value_of(joined)));
        threads.emplace_back([&, slot, i] {
            REQUIRE(slot->wait(never));
            const std::lock_guard<std::mutex> lock(mutex);
            order.push_back(i);
        });
    }
    std::this_thread::sleep_for(50ms);
    first.reset();
    for (std::thread& t : threads) {
        t.join();
    }
    CHECK(order == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE("admission: a request that leaves the line is skipped", "[admission]") {
    AdmissionQueue queue(4);
    std::optional<AdmissionQueue::Slot> running = queue.try_join();
    REQUIRE(value_of(running).wait(never));
    std::optional<AdmissionQueue::Slot> gone = queue.try_join();
    std::optional<AdmissionQueue::Slot> next = queue.try_join();
    gone.reset();  // left before its turn
    CHECK(queue.waiting() == 1);
    running.reset();
    CHECK(value_of(next).wait(never));
}

TEST_CASE("admission: give_up and close stop waiting", "[admission]") {
    AdmissionQueue queue(4);
    std::optional<AdmissionQueue::Slot> running = queue.try_join();
    REQUIRE(value_of(running).wait(never));

    std::optional<AdmissionQueue::Slot> impatient = queue.try_join();
    CHECK_FALSE(value_of(impatient).wait([] { return true; }));
    impatient.reset();

    std::optional<AdmissionQueue::Slot> waiting = queue.try_join();
    std::atomic<bool> result{true};
    std::thread waiter([&] { result = value_of(waiting).wait(never); });
    std::this_thread::sleep_for(50ms);
    queue.close();
    waiter.join();
    CHECK_FALSE(result);
    CHECK_FALSE(queue.try_join());  // closed
}

TEST_CASE("admission: wait_idle", "[admission]") {
    AdmissionQueue queue(4);
    CHECK(queue.wait_idle(std::chrono::steady_clock::now()));
    std::optional<AdmissionQueue::Slot> running = queue.try_join();
    REQUIRE(value_of(running).wait(never));
    CHECK_FALSE(queue.wait_idle(std::chrono::steady_clock::now() + 20ms));
    std::thread finisher([&] {
        std::this_thread::sleep_for(20ms);
        running.reset();
    });
    CHECK(queue.wait_idle(std::chrono::steady_clock::now() + 5s));
    finisher.join();
}

TEST_CASE("rate limiter: token bucket per client", "[admission]") {
    lisa::RateLimiter limiter(60, 2);  // one per second, bursts of two
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(limiter.acquire("a", t0) == 0);
    CHECK(limiter.acquire("a", t0) == 0);
    CHECK(limiter.acquire("a", t0) == 1);  // empty: retry in a second
    CHECK(limiter.acquire("b", t0) == 0);  // other clients are independent
    CHECK(limiter.acquire("a", t0 + 500ms) == 1);
    CHECK(limiter.acquire("a", t0 + 1500ms) == 0);
    CHECK(limiter.acquire("a", t0 + 1500ms) == 1);
    CHECK(limiter.acquire("a", t0 + 1h) == 0);  // never more than the burst
    CHECK(limiter.acquire("a", t0 + 1h) == 0);
    CHECK(limiter.acquire("a", t0 + 1h) == 1);
}

TEST_CASE("rate limiter: disabled at zero", "[admission]") {
    lisa::RateLimiter limiter(0, 1);
    CHECK_FALSE(limiter.enabled());
    for (int i = 0; i < 100; ++i) {
        CHECK(limiter.acquire("a") == 0);
    }
}
