#pragma once

#include "model.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <ggml.h>
#include <vector>
#include <string>
#include <mutex>
#include <random>

class InferenceEngine {
public:
    InferenceEngine(Model& model, const Config& cfg);
    ~InferenceEngine();
    
    std::vector<std::string> generate(const std::string& prompt,
                                      int max_new_tokens,
                                      float temperature,
                                      float top_p);
    
private:
    Model& model_;
    const Config& cfg_;
    BPETokenizer tokenizer_;
    std::mt19937 rng_;
    std::mutex mutex_;
    
    struct KVCache {
        std::vector<ggml_tensor*> k;
        std::vector<ggml_tensor*> v;
        int current_len = 0;
        void init(ggml_context* ctx, int n_layer, int n_ctx, int n_head, int n_embd);
        void clear();
    } kv_cache_;
    
    void build_graph(ggml_context* ctx, const std::vector<int>& input_ids, int pos_start, int n_tokens);
    int sample(const std::vector<float>& logits, float temperature, float top_p);
};
