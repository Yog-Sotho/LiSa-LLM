#include "stop.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>

TEST_CASE("StopMatcher passes text through without stop sequences", "[stop]") {
    lisa::StopMatcher stop({});
    CHECK(stop.push("hello") == "hello");
    CHECK_FALSE(stop.stopped());
    CHECK(stop.flush().empty());
}

TEST_CASE("StopMatcher stops at a sequence inside one piece", "[stop]") {
    lisa::StopMatcher stop({"END"});
    CHECK(stop.push("abc END def") == "abc ");
    CHECK(stop.stopped());
    CHECK(stop.push("more").empty());
    CHECK(stop.flush().empty());
}

TEST_CASE("StopMatcher holds back a possible prefix across pieces", "[stop]") {
    lisa::StopMatcher stop({"\n\nUser:"});
    CHECK(stop.push("Hi") == "Hi");
    CHECK(stop.push("\n").empty());     // could start the stop sequence
    CHECK(stop.push("\nUs").empty());   // still could
    CHECK(stop.push("er: x").empty());  // complete: nothing after it is emitted
    CHECK(stop.stopped());
}

TEST_CASE("StopMatcher releases held text once the prefix is ruled out", "[stop]") {
    lisa::StopMatcher stop({"STOP"});
    CHECK(stop.push("ST").empty());
    CHECK(stop.push("AY") == "STAY");
    CHECK(stop.push("S").empty());
    CHECK(stop.flush() == "S");  // generation ended: the partial match was just text
    CHECK_FALSE(stop.stopped());
}

TEST_CASE("StopMatcher picks the earliest of several sequences", "[stop]") {
    lisa::StopMatcher stop({"bb", "a"});
    CHECK(stop.push("xxbbya") == "xx");
    CHECK(stop.stopped());
}

TEST_CASE("StopMatcher handles multi-byte stop sequences", "[stop]") {
    lisa::StopMatcher stop({"\xE2\x80\x94"});  // em dash
    CHECK(stop.push("one \xE2\x80") == "one ");
    CHECK(stop.push("\x94 two").empty());
    CHECK(stop.stopped());
}
