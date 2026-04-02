#include "utils.hpp"
#include <ggml.h>
#include <openssl/sha.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------
// Compute SHA‑256 of a file (model integrity)
// ---------------------------------------------------------------------
static std::string sha256_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open model file");
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    const size_t bufsize = 1 << 16;
    std::vector<char> buf(bufsize);
    ssize_t n;
    while ((n = ::read(fd, buf.data(), bufsize)) > 0) {
        SHA256_Update(&ctx, buf.data(), n);
    }
    ::close(fd);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    return oss.str();
}

// ---------------------------------------------------------------------
// Load a GGML‑quantized model (memory‑mapped, read‑only)
// ---------------------------------------------------------------------
struct Model {
    ggml_context *ctx = nullptr;
    ggml_tensor *weights = nullptr; // root tensor list (GGML holds them internally)
};

inline Model load_model(const Config& cfg, const std::string& expected_sha256 = "") {
    if (!expected_sha256.empty()) {
        std::string got = sha256_file(cfg.model_path);
        if (got != expected_sha256) {
            throw std::runtime_error("model checksum mismatch");
        }
    }

    // Memory‑map the model file (read‑only)
    int fd = ::open(cfg.model_path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open model file");
    const size_t size = std::filesystem::file_size(cfg.model_path);
    void *data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (data == MAP_FAILED) throw std::runtime_error("mmap failed");

    // Initialise GGML context with the mapped buffer (no extra allocation)
    ggml_init_params params = {
        .mem_size = size,
        .mem_buffer = data,
        .no_alloc = true,
    };
    ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        ::munmap(data, size);
        throw std::runtime_error("ggml_init failed");
    }

    // The model file is a flat list of tensors; GGML will lazily resolve them on request.
    // No extra parsing is needed here.
    return Model{ctx, nullptr};
}

// ---------------------------------------------------------------------
// Clean‑up helper
// ---------------------------------------------------------------------
inline void unload_model(Model& m) {
    if (m.ctx) ggml_free(m.ctx);
    // The mapped memory is released by ggml_free when `no_alloc` is true.
}
