// libFuzzer target: OpenAI request bodies, the first thing network input reaches.
// Any outcome other than a parsed request or a RequestError (400) is a bug: another
// exception type would turn into a 500, and memory errors are caught by ASan/UBSan.

#include "request.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static const lisa::Config cfg = [] {
        lisa::Config c;
        c.model_path = "fuzz.gguf";
        c.max_context = 4096;
        return c;
    }();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): libFuzzer's byte buffer
    const std::string_view body(reinterpret_cast<const char*>(data), size);
    try {
        (void)lisa::parse_completion_request(body, cfg);
    } catch (const lisa::RequestError&) {
    }
    try {
        (void)lisa::parse_chat_request(body, cfg);
    } catch (const lisa::RequestError&) {
    }
    return 0;
}
