#pragma once

#include "model.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <ggml.h>

class InferenceEngine {
public:
    InferenceEngine(Model& model, const Config& cfg);
    ~InferenceEngine();
    
    // Thread‑safe generation
    std::vector<std::string> generate(const std::string& prompt,
                                      int max_new_tokens,
                                      float temperature,
                                      float top_p);
    
private:
    Model& model_;
    const Config& cfg_;
    BPETokenizer tokenizer_;
    std::mutex mutex_;
    
    // KV cache: for each layer, a pair of tensors (K, V) of shape [n_ctx, n_head, head_dim]
    struct KVCache {
        std::vector<ggml_tensor*> k;
        std::vector<ggml_tensor*> v;
        int current_len = 0;
    };
    KVCache kv_cache_;
    
    void init_kv_cache();
    void update_kv_cache(int layer, ggml_tensor* k, ggml_tensor* v);
    void clear_kv_cache();
    
    // Helper to apply RoPE to Q and K
    void apply_rope(ggml_tensor* q, ggml_tensor* k, int pos, int n_embd_head);
};
