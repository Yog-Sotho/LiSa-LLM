#include "auth.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

// The optional's value; throws (failing the test) when it is empty.
template <typename T>
T& value_of(std::optional<T>& optional) {
    if (!optional.has_value()) {
        throw std::logic_error("expected a value");
    }
    return *optional;
}

}  // namespace

TEST_CASE("sha256_hex", "[auth]") {
    CHECK(lisa::sha256_hex("abc") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(lisa::sha256_hex("") ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("KeyStore accepts keys and sha256: digests", "[auth]") {
    const std::string digest = lisa::sha256_hex("hashed-key-0123456789");
    const lisa::KeyStore keys({"plain-key-0123456789", "sha256:" + digest});
    REQUIRE(keys.size() == 2);

    auto plain = keys.authenticate("plain-key-0123456789");
    REQUIRE(plain);
    CHECK(value_of(plain) == "key-" + lisa::sha256_hex("plain-key-0123456789").substr(0, 8));

    auto hashed = keys.authenticate("hashed-key-0123456789");
    REQUIRE(hashed);
    CHECK(value_of(hashed) == "key-" + digest.substr(0, 8));
    CHECK(value_of(hashed) != value_of(plain));

    CHECK_FALSE(keys.authenticate(""));
    CHECK_FALSE(keys.authenticate("plain-key-012345678"));
    CHECK_FALSE(keys.authenticate("sha256:" + digest));  // the digest is not the key
}

TEST_CASE("KeyStore digests are case-insensitive hex", "[auth]") {
    std::string digest = lisa::sha256_hex("k");
    std::transform(digest.begin(), digest.end(), digest.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    CHECK(lisa::KeyStore({"sha256:" + digest}).authenticate("k"));
}

TEST_CASE("KeyStore rejects malformed digests", "[auth]") {
    CHECK_THROWS_AS(lisa::KeyStore({"sha256:abc"}), std::invalid_argument);
    CHECK_THROWS_AS(lisa::KeyStore({"sha256:" + std::string(64, 'g')}), std::invalid_argument);
}

TEST_CASE("KeyStore::from_config merges api_key and api_keys", "[auth]") {
    lisa::Config cfg;
    CHECK(lisa::KeyStore::from_config(cfg).empty());
    cfg.api_key = "one";
    cfg.api_keys = {"two", "three"};
    const lisa::KeyStore keys = lisa::KeyStore::from_config(cfg);
    CHECK(keys.size() == 3);
    CHECK(keys.authenticate("one"));
    CHECK(keys.authenticate("three"));
}
