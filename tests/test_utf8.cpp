#include "utf8.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>

TEST_CASE("Utf8Assembler passes ASCII through", "[utf8]") {
    lisa::Utf8Assembler utf8;
    CHECK(utf8.push("hello") == "hello");
    CHECK(utf8.flush().empty());
}

TEST_CASE("Utf8Assembler holds back split multi-byte characters", "[utf8]") {
    lisa::Utf8Assembler utf8;

    SECTION("2-byte (é = C3 A9)") {
        CHECK(utf8.push("caf\xC3") == "caf");
        CHECK(utf8.push("\xA9!") == "\xC3\xA9!");
    }
    SECTION("3-byte (€ = E2 82 AC) split twice") {
        CHECK(utf8.push("\xE2").empty());
        CHECK(utf8.push("\x82").empty());
        CHECK(utf8.push("\xAC") == "\xE2\x82\xAC");
    }
    SECTION("4-byte (😀 = F0 9F 98 80)") {
        CHECK(utf8.push("a\xF0\x9F") == "a");
        CHECK(utf8.push("\x98\x80") == "\xF0\x9F\x98\x80");
    }
    CHECK(utf8.flush().empty());
}

TEST_CASE("Utf8Assembler flush returns incomplete tails", "[utf8]") {
    lisa::Utf8Assembler utf8;
    CHECK(utf8.push("x\xE2\x82") == "x");
    CHECK(utf8.flush() == "\xE2\x82");
    CHECK(utf8.flush().empty());
}

TEST_CASE("Utf8Assembler passes invalid bytes through instead of stalling", "[utf8]") {
    lisa::Utf8Assembler utf8;
    CHECK(utf8.push("\xFF") == "\xFF");                                  // invalid lead byte
    CHECK(utf8.push("\x80\x80\x80\x80\x80") == "\x80\x80\x80\x80\x80");  // stray continuations
}
