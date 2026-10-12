#include "model_file.hpp"

#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace lisa {

namespace {

constexpr std::size_t kHashChunkBytes = 1U << 20;

struct DigestContextDeleter {
    void operator()(EVP_MD_CTX* ctx) const noexcept { EVP_MD_CTX_free(ctx); }
};

}  // namespace

ModelFile::ModelFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        throw std::runtime_error("cannot open model '" + path + "': " + std::strerror(errno));
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("model '" + path + "' is not a regular file");
    }
}

ModelFile::~ModelFile() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

std::string ModelFile::proc_path() const {
    return "/proc/self/fd/" + std::to_string(fd_);
}

std::string ModelFile::sha256() const {
    const std::unique_ptr<EVP_MD_CTX, DigestContextDeleter> ctx(EVP_MD_CTX_new());
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("cannot initialise SHA-256");
    }
    std::vector<unsigned char> buffer(kHashChunkBytes);
    off_t offset = 0;
    while (true) {
        const ssize_t n = ::pread(fd_, buffer.data(), buffer.size(), offset);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("cannot read model: ") + std::strerror(errno));
        }
        if (n == 0) {
            break;
        }
        if (EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(n)) != 1) {
            throw std::runtime_error("SHA-256 update failed");
        }
        offset += n;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &length) != 1) {
        throw std::runtime_error("SHA-256 finalisation failed");
    }
    static constexpr std::array<char, 16> kHex = {'0', '1', '2', '3', '4', '5', '6', '7',
                                                  '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string hex;
    hex.reserve(static_cast<std::size_t>(length) * 2);
    for (unsigned int i = 0; i < length; ++i) {
        hex.push_back(kHex[digest[i] >> 4U]);
        hex.push_back(kHex[digest[i] & 0x0FU]);
    }
    return hex;
}

void ModelFile::verify_sha256(const std::string& expected_hex) const {
    std::string expected = expected_hex;
    std::transform(expected.begin(), expected.end(), expected.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const std::string actual = sha256();
    if (actual != expected) {
        throw std::runtime_error("model SHA-256 mismatch: expected " + expected + ", got " +
                                 actual);
    }
}

}  // namespace lisa
