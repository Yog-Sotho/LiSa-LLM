#pragma once

#include <ggml.h>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

// -----------------------------------------------------------------------------
// Tensor storage and model hyperparameters
// -----------------------------------------------------------------------------
struct ModelHyperParams {
    int n_vocab = 0;
    int n_embd = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_ctx = 2048;
    float eps = 1e-5f;
    float rope_theta = 10000.0f;
};

struct Model {
    ggml_context* ctx = nullptr;
    ModelHyperParams hparams;
    std::vector<ggml_tensor*> tensors;  // all tensors by name (user can index)
    void* mapped_data = nullptr;        // for memory mapping
    size_t mapped_size = 0;
    
    ~Model();
    Model() = default;
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    
    // Disable copy
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    
    // Helper to get a tensor by name
    ggml_tensor* get_tensor(const std::string& name) const;
};

// Load model from a custom binary format (see load_model implementation)
Model load_model(const std::string& path, const std::string& expected_sha256 = "");
void unload_model(Model& model);
