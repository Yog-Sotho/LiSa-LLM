#pragma once

#include "utils.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lisa {

// Lowercase hex SHA-256 of `data`.
[[nodiscard]] std::string sha256_hex(std::string_view data);

// The accepted API keys, held only as SHA-256 digests. Entries are either a key or
// "sha256:<hex digest>", so key files need not contain the keys themselves.
class KeyStore {
public:
    KeyStore() = default;
    explicit KeyStore(const std::vector<std::string>& entries);

    // api_key plus api_keys (which load_config has merged with api_keys_file and
    // LISA_API_KEYS).
    [[nodiscard]] static KeyStore from_config(const Config& cfg);

    // True when no key is configured: authentication is disabled.
    [[nodiscard]] bool empty() const noexcept { return keys_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return keys_.size(); }

    // The id of the matching key ("key-" + 8 hex digits of its digest; safe to log), or
    // std::nullopt. Every stored digest is compared in constant time.
    [[nodiscard]] std::optional<std::string> authenticate(std::string_view key) const;

private:
    static constexpr std::size_t kDigestBytes = 32;
    using Digest = std::array<unsigned char, kDigestBytes>;

    struct Entry {
        Digest digest{};
        std::string id;
    };

    [[nodiscard]] static Digest digest_of(std::string_view data);
    [[nodiscard]] static Entry make_entry(const Digest& digest);

    std::vector<Entry> keys_;
};

}  // namespace lisa
