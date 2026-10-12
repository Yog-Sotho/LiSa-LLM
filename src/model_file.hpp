#pragma once

#include <string>

namespace lisa {

// An open, read-only model file. The file is opened once and every later use (hashing,
// loading, handing it to the sandboxed worker) goes through the same descriptor, so the
// verified file is the one that gets loaded even if the path is swapped afterwards.
class ModelFile {
public:
    // Opens `path` (must be a regular file). Throws std::runtime_error.
    explicit ModelFile(const std::string& path);
    ~ModelFile();

    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ModelFile(ModelFile&&) = delete;
    ModelFile& operator=(ModelFile&&) = delete;

    [[nodiscard]] int fd() const noexcept { return fd_; }

    // "/proc/self/fd/<n>": a path llama.cpp can open that always resolves to this file.
    [[nodiscard]] std::string proc_path() const;

    // Lower-case hex SHA-256 of the whole file.
    [[nodiscard]] std::string sha256() const;

    // Throws std::runtime_error unless sha256() equals `expected_hex` (case-insensitive).
    void verify_sha256(const std::string& expected_hex) const;

private:
    int fd_ = -1;
};

}  // namespace lisa
