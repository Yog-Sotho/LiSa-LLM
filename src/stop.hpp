#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace lisa {

// Applies stop sequences to streamed text. Text that might be the start of a stop
// sequence is held back until it is ruled out, so a stop sequence is never emitted
// even when it spans several tokens.
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops);

    // Appends `text` and returns what can be emitted now. Once a stop sequence is found,
    // returns the text before it, sets stopped() and discards everything after it.
    [[nodiscard]] std::string push(std::string_view text);

    // Returns held-back text when generation ends without a stop sequence.
    [[nodiscard]] std::string flush();

    [[nodiscard]] bool stopped() const noexcept { return stopped_; }

private:
    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_ = false;
};

}  // namespace lisa
