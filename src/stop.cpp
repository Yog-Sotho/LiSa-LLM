#include "stop.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

namespace lisa {

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    std::erase_if(stops_, [](const std::string& s) { return s.empty(); });
}

std::string StopMatcher::push(std::string_view text) {
    if (stopped_) {
        return {};
    }
    pending_.append(text);
    if (stops_.empty()) {
        return std::exchange(pending_, {});
    }

    // Earliest complete stop sequence wins.
    const std::size_t first = std::accumulate(stops_.begin(), stops_.end(), std::string::npos,
                                              [this](std::size_t best, const std::string& stop) {
                                                  return std::min(best, pending_.find(stop));
                                              });
    if (first != std::string::npos) {
        stopped_ = true;
        std::string before = pending_.substr(0, first);
        pending_.clear();
        return before;
    }

    // Hold back the longest suffix that is a prefix of some stop sequence.
    std::size_t hold = 0;
    for (const std::string& stop : stops_) {
        const std::size_t max_len = std::min(stop.size() - 1, pending_.size());
        for (std::size_t len = max_len; len > hold; --len) {
            if (std::string_view(pending_).substr(pending_.size() - len) ==
                std::string_view(stop).substr(0, len)) {
                hold = len;
                break;
            }
        }
    }
    std::string ready = pending_.substr(0, pending_.size() - hold);
    pending_.erase(0, pending_.size() - hold);
    return ready;
}

std::string StopMatcher::flush() {
    return std::exchange(pending_, {});
}

}  // namespace lisa
