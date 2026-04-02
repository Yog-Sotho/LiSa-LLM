#pragma once

#include <ggml.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

struct ModelHyperParams {
    int n_vocab = 0;
    int n_embd = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_ctx = 2048;
    float eps = 1e-5f;
    float rope_theta = 10000.0f;
};

class Model {
public:
    Model() = default;
    ~Model();
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    
    void load(const std::string& path, const std::string& expected_sha256 = "");
    void unload();
    
    ggml_tensor* get_tensor(const std::string& name) const;
    const ModelHyperParams& hparams() const { return hparams_; }
    ggml_context* ctx() const { return ctx_; }
    bool is_gpu() const { return backend_ != nullptr && ggml_backend_is_cuda(backend_); }
    ggml_backend_t backend() const { return backend_; }
    
private:
    ggml_context* ctx_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ModelHyperParams hparams_;
    std::unordered_map<std::string, ggml_tensor*> tensor_map_;
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    
    void parse_gguf(const uint8_t* data, size_t size);
    void create_backend_and_buffer();
};
