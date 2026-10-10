#include "utf8.hpp"

#include <cstddef>

namespace lisa {

namespace {

constexpr std::size_t kMaxSequenceLength = 4;

bool is_continuation(unsigned char byte) noexcept {
    return (byte & 0xC0U) == 0x80U;
}

// Length of the sequence started by `lead`; 1 for ASCII and for invalid lead bytes,
// which are passed through unchanged.
std::size_t sequence_length(unsigned char lead) noexcept {
    if ((lead & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((lead & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((lead & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 1;
}

}  // namespace

std::string Utf8Assembler::push(std::string_view bytes) {
    pending_.append(bytes);

    // Find the start of the last (possibly incomplete) sequence.
    const std::size_t size = pending_.size();
    std::size_t start = size;
    for (std::size_t back = 1; back <= kMaxSequenceLength && back <= size; ++back) {
        if (!is_continuation(static_cast<unsigned char>(pending_[size - back]))) {
            start = size - back;
            break;
        }
    }

    std::size_t cut = size;
    if (start < size &&
        start + sequence_length(static_cast<unsigned char>(pending_[start])) > size) {
        cut = start;
    }

    std::string ready = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return ready;
}

std::string Utf8Assembler::flush() {
    std::string rest;
    rest.swap(pending_);
    return rest;
}

}  // namespace lisa
