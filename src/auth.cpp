#include "auth.hpp"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <stdexcept>

namespace lisa {

namespace {

constexpr std::string_view kSha256Prefix = "sha256:";
constexpr std::string_view kHexDigits = "0123456789abcdef";
constexpr std::size_t kKeyIdHexDigits = 8;
constexpr unsigned kNibbleBits = 4;
constexpr unsigned kNibbleMask = 0x0FU;

std::string to_hex(const unsigned char* data, std::size_t size) {
    std::string hex;
    hex.reserve(size * 2);
    for (std::size_t i = 0; i < size; ++i) {
        hex += kHexDigits[data[i] >> kNibbleBits];
        hex += kHexDigits[data[i] & kNibbleMask];
    }
    return hex;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

}  // namespace

std::string sha256_hex(std::string_view data) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(data.data(), data.size(), digest, &size, EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("SHA-256 computation failed");
    }
    return to_hex(digest, size);
}

KeyStore::Digest KeyStore::digest_of(std::string_view data) {
    Digest digest{};
    unsigned int digest_size = 0;
    if (EVP_Digest(data.data(), data.size(), digest.data(), &digest_size, EVP_sha256(), nullptr) !=
            1 ||
        digest_size != kDigestBytes) {
        throw std::runtime_error("SHA-256 computation failed");
    }
    return digest;
}

KeyStore::Entry KeyStore::make_entry(const Digest& digest) {
    return Entry{digest, "key-" + to_hex(digest.data(), digest.size()).substr(0, kKeyIdHexDigits)};
}

KeyStore::KeyStore(const std::vector<std::string>& entries) {
    keys_.reserve(entries.size());
    for (const std::string& entry : entries) {
        if (!entry.starts_with(kSha256Prefix)) {
            keys_.push_back(make_entry(digest_of(entry)));
            continue;
        }
        const std::string_view hex = std::string_view(entry).substr(kSha256Prefix.size());
        if (hex.size() != kDigestBytes * 2) {
            throw std::invalid_argument("malformed sha256: API key entry");
        }
        Digest digest{};
        for (std::size_t i = 0; i < kDigestBytes; ++i) {
            const int high = hex_value(hex[2 * i]);
            const int low = hex_value(hex[2 * i + 1]);
            if (high < 0 || low < 0) {
                throw std::invalid_argument("malformed sha256: API key entry");
            }
            digest[i] = static_cast<unsigned char>((high << kNibbleBits) | low);
        }
        keys_.push_back(make_entry(digest));
    }
}

KeyStore KeyStore::from_config(const Config& cfg) {
    std::vector<std::string> entries = cfg.api_keys;
    if (!cfg.api_key.empty()) {
        entries.push_back(cfg.api_key);
    }
    return KeyStore(entries);
}

std::optional<std::string> KeyStore::authenticate(std::string_view key) const {
    // Comparing digests of equal length keeps the comparison independent of the key's
    // length and content; every entry is checked so timing doesn't reveal which matched.
    const Digest given = digest_of(key);
    const Entry* match = nullptr;
    for (const Entry& entry : keys_) {
        if (CRYPTO_memcmp(given.data(), entry.digest.data(), kDigestBytes) == 0) {
            match = &entry;
        }
    }
    if (match == nullptr) {
        return std::nullopt;
    }
    return match->id;
}

}  // namespace lisa
