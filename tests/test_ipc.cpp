#include "ipc.hpp"
#include "model_file.hpp"
#include "protocol.hpp"

#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>

namespace {

std::pair<lisa::ipc::Channel, lisa::ipc::Channel> channel_pair() {
    std::array<int, 2> sv{};
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv.data()) == 0);
    return {lisa::ipc::Channel(sv[0]), lisa::ipc::Channel(sv[1])};
}

}  // namespace

TEST_CASE("ipc: frames round-trip, including large ones", "[ipc]") {
    auto [a, b] = channel_pair();
    a.send({{"op", "tokenize"}, {"text", "héllo"}});
    CHECK(b.receive() == lisa::json{{"op", "tokenize"}, {"text", "héllo"}});

    const std::string big(3U << 20, 'x');  // larger than a socket buffer
    std::thread writer([&a, &big] { a.send({{"text", big}}); });
    CHECK(b.receive()["text"].get<std::string>().size() == big.size());
    writer.join();

    CHECK_FALSE(b.readable(0));
    a.send({{"op", "cancel"}});
    CHECK(b.readable(1000));
}

TEST_CASE("ipc: closed peers, oversized and malformed frames", "[ipc]") {
    {
        auto [a, b] = channel_pair();
        a.close();
        try {
            (void)b.receive();
            FAIL("expected IpcError");
        } catch (const lisa::ipc::IpcError& e) {
            CHECK(e.peer_closed());
        }
        CHECK_THROWS_AS(b.send({{"x", 1}}), lisa::ipc::IpcError);  // no SIGPIPE
    }
    {
        auto [a, b] = channel_pair();
        const std::array<unsigned char, 4> huge = {0x7F, 0xFF, 0xFF, 0xFF};
        REQUIRE(::write(a.fd(), huge.data(), huge.size()) == 4);
        CHECK_THROWS_WITH(b.receive(), Catch::Matchers::ContainsSubstring("size limit"));
    }
    {
        auto [a, b] = channel_pair();
        const std::string frame = std::string("\0\0\0\x03", 4) + "[1]";
        REQUIRE(::write(a.fd(), frame.data(), frame.size()) == static_cast<ssize_t>(frame.size()));
        CHECK_THROWS_WITH(b.receive(), Catch::Matchers::ContainsSubstring("malformed"));
    }
}

TEST_CASE("protocol: values survive serialisation", "[ipc]") {
    lisa::GenerationParams params;
    params.max_tokens = 17;
    params.max_tokens_explicit = true;
    params.stop = {"\n", "END"};
    params.sampling.temperature = 0.3f;
    params.sampling.top_k = 40;
    params.sampling.seed = 99;
    params.stream = true;
    const auto back = lisa::json(params).get<lisa::GenerationParams>();
    CHECK(back.max_tokens == 17);
    CHECK(back.stop == params.stop);
    CHECK(back.sampling.temperature == 0.3f);
    CHECK(back.sampling.top_k == 40);
    CHECK(back.sampling.seed == 99U);
    CHECK(back.stream);

    const lisa::GenerationResult result{5, 2, 3, lisa::StopReason::stop_sequence};
    const auto r = lisa::json(result).get<lisa::GenerationResult>();
    CHECK(r.reason == lisa::StopReason::stop_sequence);
    CHECK(r.completion_tokens == 3);
    CHECK(r.cached_tokens == 2);

    lisa::Config cfg;
    cfg.api_key = "secret";
    cfg.chat_template = "llama3";
    const lisa::json settings = lisa::WorkerSettings::from_config(cfg);
    CHECK(settings["chat_template"] == "llama3");
    CHECK(settings.dump().find("secret") == std::string::npos);  // secrets stay in the broker
}

TEST_CASE("protocol: errors keep their kind", "[ipc]") {
    const auto request = lisa::error_reply(lisa::RequestError("bad top_p", "top_p"));
    bool caught = false;
    try {
        lisa::throw_error_reply(request);
    } catch (const lisa::RequestError& e) {
        caught = true;
        CHECK(std::string(e.what()) == "bad top_p");
        CHECK(e.param() == "top_p");
    }
    CHECK(caught);
    const auto internal = lisa::error_reply(std::runtime_error("boom"));
    CHECK_THROWS_AS(lisa::throw_error_reply(internal), std::runtime_error);
}

TEST_CASE("model file: SHA-256 and validation", "[ipc]") {
    const auto path = std::filesystem::temp_directory_path() / "lisa_model_hash_test.bin";
    std::ofstream(path) << "abc";
    const lisa::ModelFile file(path.string());
    // SHA-256("abc")
    CHECK(file.sha256() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK_NOTHROW(
        file.verify_sha256("BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"));
    CHECK_THROWS_WITH(file.verify_sha256(std::string(64, 'f')),
                      Catch::Matchers::ContainsSubstring("mismatch"));
    CHECK(file.proc_path() == "/proc/self/fd/" + std::to_string(file.fd()));
    std::filesystem::remove(path);

    CHECK_THROWS_WITH(lisa::ModelFile("/nonexistent.gguf"),
                      Catch::Matchers::ContainsSubstring("cannot open"));
    CHECK_THROWS_WITH(lisa::ModelFile("/tmp"), Catch::Matchers::ContainsSubstring("regular file"));
}
