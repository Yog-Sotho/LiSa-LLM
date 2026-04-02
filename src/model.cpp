#include "model.hpp"
#include "utils.hpp"
#include <openssl/sha.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <algorithm>

static std::string sha256_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open file for checksum");
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    char buf[65536];
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf))) > 0) SHA256_Update(&ctx, buf, n);
    ::close(fd);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    return oss.str();
}

static uint32_t read_u32(const uint8_t*& p) {
    uint32_t v = p[0] | (p[1]<<8) | (p[2]<<16) | (p[3]<<24);
    p += 4;
    return v;
}

static uint64_t read_u64(const uint8_t*& p) {
    uint64_t v = (uint64_t)p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) | ((uint64_t)p[3]<<24) |
                 ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) | ((uint64_t)p[6]<<48) | ((uint64_t)p[7]<<56);
    p += 8;
    return v;
}

static std::string read_string(const uint8_t*& p) {
    uint64_t len = read_u64(p);
    std::string s((const char*)p, len);
    p += len;
    return s;
}

void Model::create_backend_and_buffer() {
#ifdef GGML_USE_CUDA
    if (backend_ == nullptr) {
        backend_ = ggml_backend_cuda_init(0);
        if (!backend_) throw std::runtime_error("failed to initialize CUDA backend");
    }
#else
    backend_ = ggml_backend_cpu_init();
#endif
    size_t ctx_size = ggml_get_mem_size(ctx_);
    buffer_ = ggml_backend_alloc_buffer(backend_, ctx_size);
    if (!buffer_) throw std::runtime_error("failed to allocate backend buffer");
    ggml_backend_buffer_clear(buffer_, 0);
    // Copy tensors from CPU context to backend buffer
    for (auto& pair : tensor_map_) {
        ggml_tensor* cpu_tensor = pair.second;
        ggml_tensor* dev_tensor = ggml_dup_tensor(ctx_, cpu_tensor);
        dev_tensor->data = ggml_backend_buffer_get_base(buffer_) + (size_t)dev_tensor->data;
        ggml_backend_tensor_set(dev_tensor, cpu_tensor->data, 0, ggml_nbytes(cpu_tensor));
        pair.second = dev_tensor;
    }
}

void Model::parse_gguf(const uint8_t* data, size_t size) {
    // Minimal GGUF parser (spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
    const uint8_t* p = data;
    if (read_u32(p) != 0x46554747) throw std::runtime_error("invalid GGUF magic");
    if (read_u32(p) != 3) throw std::runtime_error("unsupported GGUF version");
    uint64_t tensor_count = read_u64(p);
    uint64_t kv_count = read_u64(p);
    
    // Read key-value pairs to extract hyperparameters
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = read_string(p);
        uint32_t type = read_u32(p);
        if (key == "general.architecture") {
            std::string arch = read_string(p);
            if (arch != "llama") throw std::runtime_error("only llama architecture supported");
        } else if (key == "llama.context_length") {
            hparams_.n_ctx = read_u64(p);
        } else if (key == "llama.embedding_length") {
            hparams_.n_embd = read_u64(p);
        } else if (key == "llama.block_count") {
            hparams_.n_layer = read_u64(p);
        } else if (key == "llama.attention_head_count") {
            hparams_.n_head = read_u64(p);
        } else if (key == "llama.vocab_size") {
            hparams_.n_vocab = read_u64(p);
        } else if (key == "llama.rope.freq_base") {
            hparams_.rope_theta = read_u64(p);
        } else if (key == "llama.layer_norm_eps") {
            hparams_.eps = read_u32(p);  // float stored as uint32
        } else {
            // skip value
            uint32_t val_type = type;
            switch (val_type) {
                case 4: read_u64(p); break; // uint64
                case 5: p += 4; break;      // float32
                case 6: p += 8; break;      // bool
                default: throw std::runtime_error("unknown kv type");
            }
        }
    }
    
    // Create GGML context
    size_t ctx_size = 1024 * 1024 * 200; // 200 MB, will grow
    ggml_init_params params = { .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true };
    ctx_ = ggml_init(params);
    if (!ctx_) throw std::runtime_error("ggml_init failed");
    
    // Read tensors
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name = read_string(p);
        uint32_t n_dims = read_u32(p);
        std::vector<int64_t> dims(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j) dims[j] = read_u64(p);
        uint32_t type = read_u32(p);
        uint64_t offset = read_u64(p);
        ggml_type gtype = (ggml_type)type;
        ggml_tensor* tensor = nullptr;
        if (n_dims == 1) tensor = ggml_new_tensor_1d(ctx_, gtype, dims[0]);
        else if (n_dims == 2) tensor = ggml_new_tensor_2d(ctx_, gtype, dims[0], dims[1]);
        else if (n_dims == 3) tensor = ggml_new_tensor_3d(ctx_, gtype, dims[0], dims[1], dims[2]);
        else if (n_dims == 4) tensor = ggml_new_tensor_4d(ctx_, gtype, dims[0], dims[1], dims[2], dims[3]);
        else throw std::runtime_error("unsupported number of dims");
        
        // Copy data from mmap (tensor data starts at `offset` from end of header)
        const uint8_t* tensor_data = data + offset;
        size_t nbytes = ggml_nbytes(tensor);
        memcpy(tensor->data, tensor_data, nbytes);
        tensor_map_[name] = tensor;
    }
}

void Model::load(const std::string& path, const std::string& expected_sha256) {
    if (!expected_sha256.empty()) {
        std::string got = sha256_file(path);
        if (got != expected_sha256) throw std::runtime_error("model checksum mismatch");
    }
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open model file");
    mapped_size_ = std::filesystem::file_size(path);
    mapped_data_ = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped_data_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    
    parse_gguf((const uint8_t*)mapped_data_, mapped_size_);
    if (backend_ == nullptr) create_backend_and_buffer();
}

void Model::unload() {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    if (mapped_data_) {
        munmap(mapped_data_, mapped_size_);
        mapped_data_ = nullptr;
    }
    tensor_map_.clear();
}

Model::~Model() { unload(); }
Model::Model(Model&& other) noexcept
    : ctx_(other.ctx_), backend_(other.backend_), buffer_(other.buffer_),
      hparams_(other.hparams_), tensor_map_(std::move(other.tensor_map_)),
      mapped_data_(other.mapped_data_), mapped_size_(other.mapped_size_) {
    other.ctx_ = nullptr; other.backend_ = nullptr; other.buffer_ = nullptr;
    other.mapped_data_ = nullptr;
}
Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        unload();
        ctx_ = other.ctx_; backend_ = other.backend_; buffer_ = other.buffer_;
        hparams_ = other.hparams_; tensor_map_ = std::move(other.tensor_map_);
        mapped_data_ = other.mapped_data_; mapped_size_ = other.mapped_size_;
        other.ctx_ = nullptr; other.backend_ = nullptr; other.buffer_ = nullptr;
        other.mapped_data_ = nullptr;
    }
    return *this;
}

ggml_tensor* Model::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
