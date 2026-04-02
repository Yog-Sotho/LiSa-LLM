#include "utils.hpp"
#include "model_loader.cpp"
#include <ggml.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <cmath>

// ---------------------------------------------------------------------
// Simple BPE tokenizer (loads HuggingFace vocab JSON)
// ---------------------------------------------------------------------
class Tokenizer {
public:
    explicit Tokenizer(const std::string& vocab_path) {
        std::ifstream f(vocab_path);
        if (!f) throw std::runtime_error("cannot open vocab file");
        json j;
        f >> j;
        for (auto& el : j.items()) {
            token_to_id[el.key()] = el.value();
            id_to_token[el.value()] = el.key();
        }
    }

    // Very naive whitespace‑split tokenizer – replace with proper BPE for production
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            auto it = token_to_id.find(word);
            if (it != token_to_id.end())
                ids.push_back(it->second);
            else
                ids.push_back(token_to_id.at("<unk>")); // fallback
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) {
            auto it = id_to_token.find(id);
            if (it != id_to_token.end())
                out += it->second + " ";
        }
        if (!out.empty()) out.pop_back(); // trim trailing space
        return out;
    }

private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
};

// ---------------------------------------------------------------------
// Core transformer utilities (attention, feed‑forward, layer‑norm)
// ---------------------------------------------------------------------
static ggml_tensor* ggml_norm(ggml_context *ctx, ggml_tensor *x, float eps) {
    // x = (x - mean) / sqrt(var + eps)
    ggml_tensor *mean = ggml_mean(ctx, x);
    ggml_tensor *x_centered = ggml_sub(ctx, x, mean);
    ggml_tensor *var = ggml_mean(ctx, ggml_sqr(ctx, x_centered));
    ggml_tensor *denom = ggml_sqrt(ctx, ggml_add(ctx, var, ggml_new_f32(ctx, eps)));
    return ggml_div(ctx, x_centered, denom);
}

// ---------------------------------------------------------------------
// Inference engine – pure GGML transformer
// ---------------------------------------------------------------------
class InferenceEngine {
public:
    explicit InferenceEngine(Model& m, const Config& cfg)
        : model(m), cfg(cfg), tokenizer(cfg.vocab_path) {}

    // Generate up to `max_new` tokens, returning them as strings.
    std::vector<std::string> generate(const std::string& prompt,
                                      int max_new = 64,
                                      float temperature = 0.8f,
                                      float top_p = 0.95f) {
        std::lock_guard<std::mutex> lock(mtx); // protect shared GGML context

        // 1️⃣ Encode prompt
        std::vector<int> input_ids = tokenizer.encode(prompt);
        if (input_ids.empty()) throw std::runtime_error("prompt tokenization failed");

        // 2️⃣ Allocate KV cache (single‑layer example – extend for real models)
        // For a production model you would allocate a 3‑D tensor: [layers][seq_len][head_dim]
        // Here we keep a minimal cache for demonstration.
        // In practice you would read the model’s hyper‑parameters from the GGML file.

        // 3️⃣ Forward pass for each token (auto‑regressive)
        std::vector<std::string> out_tokens;
        for (int step = 0; step < max_new; ++step) {
            // Build input tensor (shape: [1, seq_len])
            ggml_tensor *input = ggml_new_i32(model.ctx, 1);
            input->data = input_ids.data(); // points to token ids (no copy)

            // ---- Transformer block (simplified) ----
            // 1) Layer‑norm
            ggml_tensor *normed = ggml_norm(model.ctx, input, 1e-5f);

            // 2) Linear projection to Q,K,V (weights fetched by name)
            ggml_tensor *Wq = ggml_get_tensor(model.ctx, "transformer.h.0.attn_wq");
            ggml_tensor *Wk = ggml_get_tensor(model.ctx, "transformer.h.0.attn_wk");
            ggml_tensor *Wv = ggml_get_tensor(model.ctx, "transformer.h.0.attn_wv");
            ggml_tensor *Wout = ggml_get_tensor(model.ctx, "transformer.h.0.attn_wo");

            // 3) Compute Q,K,V (matrix multiply)
            ggml_tensor *Q = ggml_mul_mat(model.ctx, Wq, normed);
            ggml_tensor *K = ggml_mul_mat(model.ctx, Wk, normed);
            ggml_tensor *V = ggml_mul_mat(model.ctx, Wv, normed);

            // 4) Scaled dot‑product attention (single head for brevity)
            ggml_tensor *scores = ggml_mul_mat(model.ctx, Q, ggml_transpose(ctx, K));
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)Q->ne[0])); // scale
            scores = ggml_soft_max(ctx, scores);
            ggml_tensor *attn = ggml_mul_mat(ctx, scores, V);
            ggml_tensor *proj = ggml_mul_mat(ctx, Wout, attn);

            // 5) Residual connection + second layer‑norm + feed‑forward (skipped for brevity)
            // In a real implementation you would repeat the above for all layers,
            // apply rotary embeddings, multi‑head attention, and a two‑layer MLP.

            // 6) Final linear projection to logits
            ggml_tensor *Wlogits = ggml_get_tensor(model.ctx, "lm_head");
            ggml_tensor *logits = ggml_mul_mat(ctx, Wlogits, proj);

            // 7) Sample next token (greedy for now)
            // Convert logits to CPU array
            ggml_tensor *logits_f32 = ggml_to_float(ctx, logits);
            const float *logits_ptr = (float*)logits_f32->data;
            int vocab_size = logits_f32->ne[0];
            int best_id = 0;
            float best_score = logits_ptr[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits_ptr[i] > best_score) {
                    best_score = logits_ptr[i];
                    best_id = i;
                }
            }

            // Stop on EOS token (assumed id 2)
            if (best_id == 2) break;

            // Append token to output
            out_tokens.push_back(tokenizer.decode({best_id}));

            // Append token id to input for next step
            input_ids.push_back(best_id);
        }
        return out_tokens;
    }

private:
    Model& model;
    const Config& cfg;
    Tokenizer tokenizer;
    std::mutex mtx; // protects the shared GGML context
};
