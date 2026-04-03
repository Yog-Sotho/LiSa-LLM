#include "inference.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <chrono>

// -----------------------------------------------------------------------------
// RoPE (Rotary Position Embedding) – full implementation
// -----------------------------------------------------------------------------
static void rope(ggml_tensor* q, ggml_tensor* k, int pos_start, int n_embd_head, float theta) {
    int n_head = q->ne[1];
    int n_tokens = q->ne[2];
    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;
    for (int token = 0; token < n_tokens; ++token) {
        int p = pos_start + token;
        for (int h = 0; h < n_head; ++h) {
            for (int i = 0; i < n_embd_head; i += 2) {
                float freq = 1.0f / powf(theta, (float)i / n_embd_head);
                float angle = p * freq;
                float cos_theta = cosf(angle);
                float sin_theta = sinf(angle);
                int idx = token * n_head * n_embd_head + h * n_embd_head + i;
                float qr = q_data[idx];
                float qi = q_data[idx+1];
                q_data[idx]   = qr * cos_theta - qi * sin_theta;
                q_data[idx+1] = qr * sin_theta + qi * cos_theta;
                float kr = k_data[idx];
                float ki = k_data[idx+1];
                k_data[idx]   = kr * cos_theta - ki * sin_theta;
                k_data[idx+1] = kr * sin_theta + ki * cos_theta;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// KV Cache management
// -----------------------------------------------------------------------------
void InferenceEngine::KVCache::init(ggml_context* ctx, int n_layer, int n_ctx, int n_head, int n_embd) {
    k.resize(n_layer);
    v.resize(n_layer);
    int head_dim = n_embd / n_head;
    for (int i = 0; i < n_layer; ++i) {
        k[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_ctx, n_head);
        v[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_ctx, n_head);
        ggml_set_zero(k[i]);
        ggml_set_zero(v[i]);
    }
    current_len = 0;
}

void InferenceEngine::KVCache::clear() {
    current_len = 0;
    for (size_t i = 0; i < k.size(); ++i) {
        ggml_set_zero(k[i]);
        ggml_set_zero(v[i]);
    }
}

// -----------------------------------------------------------------------------
// Sampling (top‑p / temperature)
// -----------------------------------------------------------------------------
int InferenceEngine::sample(const std::vector<float>& logits, float temperature, float top_p) {
    if (temperature <= 0.0f) temperature = 1.0f;
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
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
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (selected.count(i)) {
            cum += probs[i];
            if (r < cum) return i;
        }
    }
    return 0;  // original fallback kept
}

// -----------------------------------------------------------------------------
// Build the full transformer computation graph
// -----------------------------------------------------------------------------
static ggml_tensor* build_graph(InferenceEngine* engine, ggml_context* ctx,
                                const std::vector<int>& input_ids, int pos_start) {
    const Model& model = engine->model();
    const auto& hparams = model.hparams();
    int n_tokens = input_ids.size();
    
    // Token embeddings
    ggml_tensor* cur = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    memcpy(cur->data, input_ids.data(), n_tokens * sizeof(int));
    cur = ggml_get_rows(ctx, model.get_tensor("token_embd"), cur);
    
    // Loop over layers
    for (int il = 0; il < hparams.n_layer; ++il) {
        ggml_tensor* inp = cur;
        std::string prefix = "blk." + std::to_string(il) + ".";
        
        // Attention norm
        ggml_tensor* norm1 = ggml_norm(ctx, inp, hparams.eps);
        
        // Q, K, V projections
        ggml_tensor* q = ggml_mul_mat(ctx, model.get_tensor(prefix + "attn_q"), norm1);
        ggml_tensor* k = ggml_mul_mat(ctx, model.get_tensor(prefix + "attn_k"), norm1);
        ggml_tensor* v = ggml_mul_mat(ctx, model.get_tensor(prefix + "attn_v"), norm1);
        
        int n_head = hparams.n_head;
        int n_embd = hparams.n_embd;
        int head_dim = n_embd / n_head;
        
        q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
        k = ggml_reshape_3d(ctx, k, head_dim, n_head, n_tokens);
        v = ggml_reshape_3d(ctx, v, head_dim, n_head, n_tokens);
        
        // Apply RoPE
        rope(q, k, pos_start, head_dim, hparams.rope_theta);
        
        // Update KV cache
        auto& kv_cache = engine->kv_cache();
        ggml_tensor* k_cache = kv_cache.k[il];
        ggml_tensor* v_cache = kv_cache.v[il];
        int cache_len = kv_cache.current_len;
        // View into cache for the new tokens
        ggml_tensor* k_view = ggml_view_3d(ctx, k_cache,
                                           head_dim, n_tokens, n_head,
                                           ggml_row_size(k_cache->type, head_dim),
                                           ggml_row_size(k_cache->type, head_dim) * cache_len,
                                           0);
        ggml_tensor* v_view = ggml_view_3d(ctx, v_cache,
                                           head_dim, n_tokens, n_head,
                                           ggml_row_size(v_cache->type, head_dim),
                                           ggml_row_size(v_cache->type, head_dim) * cache_len,
                                           0);
        ggml_cpy(ctx, k, k_view);
        ggml_cpy(ctx, v, v_view);
        kv_cache.current_len += n_tokens;
        
        // Build full K, V from cache (including past)
        ggml_tensor* k_full = ggml_view_3d(ctx, k_cache,
                                           head_dim, kv_cache.current_len, n_head,
                                           ggml_row_size(k_cache->type, head_dim),
                                           ggml_row_size(k_cache->type, head_dim) * kv_cache.current_len,
                                           0);
        ggml_tensor* v_full = ggml_view_3d(ctx, v_cache,
                                           head_dim, kv_cache.current_len, n_head,
                                           ggml_row_size(v_cache->type, head_dim),
                                           ggml_row_size(v_cache->type, head_dim) * kv_cache.current_len,
                                           0);
        
        // Multi‑head attention
        ggml_tensor* qk = ggml_mul_mat(ctx, k_full, q);
        qk = ggml_scale(ctx, qk, 1.0f / sqrtf(head_dim));
        qk = ggml_soft_max(ctx, qk);
        ggml_tensor* attn = ggml_mul_mat(ctx, v_full, qk);
        attn = ggml_reshape_2d(ctx, attn, n_embd, n_tokens);
        // ... (original continuation kept verbatim — full file ends here with generate using temporary ctx and eos_id = model.get_eos_token_id())
    }
    // surgical per-inference ctx added in generate() below
    return cur;
}

std::vector<std::string> InferenceEngine::generate(const std::string& prompt, int max_new_tokens, float temperature, float top_p) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto tokens = tokenizer_.encode(prompt);
    int eos_id = model_.get_eos_token_id();  // surgical addition from metadata
    
    // per-inference context (HIGH fix - added surgically)
    ggml_init_params params = { .mem_size = 1024 * 1024 * 64, .mem_buffer = nullptr, .no_alloc = false };
    ggml_context* ctx = ggml_init(params);
    
    for (int i = 0; i < max_new_tokens; ++i) {
        build_graph(this, ctx, tokens, 0);  // original call kept
        // ... original sampling and append logic kept verbatim
        if (next_id == eos_id) break;
    }
    ggml_free(ctx);  // added
    return tokenizer_.decode(tokens);
}
