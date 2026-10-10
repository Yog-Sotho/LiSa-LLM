#pragma once

#include <string>
#include <string_view>

namespace lisa {

// Token pieces can end in the middle of a multi-byte UTF-8 character (byte-fallback
// tokens). Utf8Assembler buffers such tails so streamed text never contains a split
// character.
class Utf8Assembler {
public:
    // Appends `bytes` and returns the longest buffered prefix that ends on a complete
    // character. Incomplete trailing bytes stay buffered.
    [[nodiscard]] std::string push(std::string_view bytes);

    // Returns everything still buffered (possibly an incomplete sequence) and resets.
    [[nodiscard]] std::string flush();

private:
    std::string pending_;
};

}  // namespace lisa
