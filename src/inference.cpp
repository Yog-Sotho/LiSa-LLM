#include "inference.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <unordered_set>

static float sample_top_p(const std::vector<float>& logits, float temperature, float top_p) {
    if (temperature <= 0) temperature = 1.0f;
    std::vector<float> probs(logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] / temperature);
        sum_exp += probs[i];
    }
    for (float& p : probs) p /= sum_exp;
    
    // Top‑p (nucleus) sampling
    std::vector<std::pair<float, int>> sorted;
    for (size_t i = 0; i < probs.size(); ++i) sorted.emplace_back(probs[i], i);
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });
    float cum = 0.0f;
    std::unordered_set<int> selected;
    for (auto& [prob, idx] : sorted) {
        cum += prob;
        selected.insert(idx);
        if (cum >= top_p) break;
    }
    
    // Sample from selected set
    float r = (float)rand() / RAND_MAX;
    cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (selected.count(i)) {
            cum += probs[i];
            if (r < cum) return i;
        }
    }
    return 0;
}

InferenceEngine::InferenceEngine(Model& model, const Config& cfg)
    : model_(model), cfg_(cfg) {
    tokenizer_.load(cfg.vocab_path, cfg.merges_path);
    init_kv_cache();
}

InferenceEngine::~InferenceEngine() = default;

void InferenceEngine::init_kv_cache() {
    int n_layer = model_.hparams.n_layer;
    int n_head = model_.hparams.n_head;
    int n_embd = model_.hparams.n_embd;
    int head_dim = n_embd / n_head;
    int n_ctx = model_.hparams.n_ctx;
    
    kv_cache_.k.resize(n_layer);
    kv_cache_.v.resize(n_layer);
    // We need a separate GGML context for the cache? GGML tensors must belong to a context.
    // For simplicity, we allocate new tensors in the model's context (which may be read‑only).
    // Better: create a separate context for cache.
    // Here we'll skip full implementation for brevity – assume we have a cache context.
}

void InferenceEngine::clear_kv_cache() {
    kv_cache_.current_len = 0;
    // Reset tensor data to zero
}

void InferenceEngine::apply_rope(ggml_tensor* q, ggml_tensor* k, int pos, int n_embd_head) {
    // Real RoPE implementation: for each head, rotate the embeddings.
    // This is a placeholder – the actual math is omitted for space.
}

std::vector<std::string> InferenceEngine::generate(const std::string& prompt,
                                                   int max_new_tokens,
                                                   float temperature,
                                                   float top_p) {
    std::lock_guard<std::mutex> lock(mutex_);
    clear_kv_cache();
    
    std::vector<int> input_ids = tokenizer_.encode(prompt);
    if (input_ids.empty()) throw std::runtime_error("empty prompt tokenization");
    
    std::vector<std::string> out_tokens;
    for (int step = 0; step < max_new_tokens; ++step) {
        // Build input tensor (batch=1, seq_len = current length)
        int seq_len = input_ids.size();
        // In a real implementation, you would create a tensor of shape [1, seq_len] with the ids.
        // Then run the full transformer forward pass (layers, attention, MLP).
        // Since this is a massive function, we provide the skeleton.
        
        // After forward pass, get logits for the last token.
        // Here we simulate random logits for demonstration.
        std::vector<float> logits(model_.hparams.n_vocab);
        for (int i = 0; i < model_.hparams.n_vocab; ++i)
            logits[i] = ((float)rand() / RAND_MAX) - 0.5f;
        
        int next_id = sample_top_p(logits, temperature, top_p);
        if (next_id == 2) break;  // EOS token id 2
        
        std::string token_str = tokenizer_.decode({next_id});
        out_tokens.push_back(token_str);
        input_ids.push_back(next_id);
        kv_cache_.current_len = input_ids.size();
    }
    return out_tokens;
}
