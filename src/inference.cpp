#include "inference.hpp"

#include "utf8.hpp"

#include <llama.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace lisa {

namespace {

constexpr std::size_t kPieceBufferSize = 64;

// Routes llama.cpp/ggml warnings and errors into the structured log; INFO/DEBUG chatter
// is dropped.
void llama_log_bridge(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level != GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_ERROR) {
        return;
    }
    std::string message(text != nullptr ? text : "");
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    if (message.empty()) {
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR) {
        log_error(message, {{"component", "llama"}});
    } else {
        log_warn(message, {{"component", "llama"}});
    }
}

void init_backend_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(llama_log_bridge, nullptr);
        llama_backend_init();
    });
}

struct SamplerDeleter {
    void operator()(llama_sampler* sampler) const noexcept { llama_sampler_free(sampler); }
};
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

SamplerPtr make_sampler(const CompletionRequest& request) {
    SamplerPtr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!chain) {
        throw std::runtime_error("failed to create sampler");
    }
    if (request.temperature <= 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(request.top_p, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(request.temperature));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    return chain;
}

}  // namespace

void InferenceEngine::ModelDeleter::operator()(llama_model* model) const noexcept {
    llama_model_free(model);
}

void InferenceEngine::ContextDeleter::operator()(llama_context* ctx) const noexcept {
    llama_free(ctx);
}

InferenceEngine::InferenceEngine(const Config& cfg) {
    init_backend_once();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.use_gpu ? cfg.gpu_layers : 0;

    model_.reset(llama_model_load_from_file(cfg.model_path.c_str(), model_params));
    if (!model_) {
        throw std::runtime_error("failed to load model '" + cfg.model_path + "'");
    }
    vocab_ = llama_model_get_vocab(model_.get());

    const std::int32_t n_ctx_train = llama_model_n_ctx_train(model_.get());
    std::uint32_t n_ctx = cfg.max_context;
    if (n_ctx_train > 0) {
        n_ctx = std::min(n_ctx, static_cast<std::uint32_t>(n_ctx_train));
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;  // the whole prompt is evaluated in one llama_decode call
    ctx_params.n_threads = cfg.n_threads;
    ctx_params.n_threads_batch = cfg.n_threads;
    ctx_params.no_perf = true;

    ctx_.reset(llama_init_from_model(model_.get(), ctx_params));
    if (!ctx_) {
        throw std::runtime_error("failed to create inference context");
    }

    std::string description(256, '\0');
    const std::int32_t len = llama_model_desc(model_.get(), description.data(), description.size());
    description.resize(len > 0 ? std::min(static_cast<std::size_t>(len), description.size()) : 0);

    info_.description = std::move(description);
    info_.n_layer = llama_model_n_layer(model_.get());
    info_.n_ctx_train = n_ctx_train;
    // llama.cpp may pad the allocation; requests are still limited to what was configured.
    info_.n_ctx = std::min(n_ctx, llama_n_ctx(ctx_.get()));
    info_.n_params = llama_model_n_params(model_.get());
    info_.gpu_offload = llama_supports_gpu_offload() && model_params.n_gpu_layers != 0;
}

InferenceEngine::~InferenceEngine() = default;

std::vector<Token> InferenceEngine::tokenize(std::string_view text) const {
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("prompt is too large");
    }
    const auto text_len = static_cast<std::int32_t>(text.size());

    // Usually enough (≤ 1 token per byte + BOS/EOS); retried with the exact size otherwise.
    std::vector<Token> tokens(text.size() + 2);
    std::int32_t n = llama_tokenize(vocab_, text.data(), text_len, tokens.data(),
                                    static_cast<std::int32_t>(tokens.size()),
                                    /*add_special=*/true, /*parse_special=*/false);
    if (n == INT32_MIN) {
        throw std::invalid_argument("prompt is too large");
    }
    if (n < 0) {
        tokens.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(vocab_, text.data(), text_len, tokens.data(),
                           static_cast<std::int32_t>(tokens.size()), true, false);
        if (n < 0) {
            throw std::invalid_argument("prompt tokenization failed");
        }
    }
    tokens.resize(static_cast<std::size_t>(n));
    if (tokens.empty()) {
        throw std::invalid_argument("prompt produced no tokens");
    }
    return tokens;
}

std::string InferenceEngine::token_to_piece(Token token) const {
    std::string piece(kPieceBufferSize, '\0');
    std::int32_t n = llama_token_to_piece(vocab_, token, piece.data(),
                                          static_cast<std::int32_t>(piece.size()), 0, false);
    if (n < 0) {
        piece.resize(static_cast<std::size_t>(-n));
        n = llama_token_to_piece(vocab_, token, piece.data(),
                                 static_cast<std::int32_t>(piece.size()), 0, false);
        if (n < 0) {
            throw std::runtime_error("failed to detokenize token");
        }
    }
    piece.resize(static_cast<std::size_t>(n));
    return piece;
}

GenerationResult InferenceEngine::generate(const std::vector<Token>& prompt_tokens,
                                           const CompletionRequest& request,
                                           const PieceCallback& on_piece) {
    if (prompt_tokens.empty() || request.max_new_tokens < 1) {
        throw std::invalid_argument("empty prompt or max_new_tokens < 1");
    }
    if (prompt_tokens.size() + static_cast<std::size_t>(request.max_new_tokens) > info_.n_ctx) {
        throw std::invalid_argument("prompt plus max_new_tokens exceeds the context window of " +
                                    std::to_string(info_.n_ctx) + " tokens");
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    llama_context* ctx = ctx_.get();
    llama_memory_clear(llama_get_memory(ctx), true);
    const SamplerPtr sampler = make_sampler(request);

    GenerationResult result;
    result.prompt_tokens = prompt_tokens.size();

    // llama_batch takes a non-const pointer; it does not modify the tokens.
    std::vector<Token> prompt = prompt_tokens;
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(),
                                              static_cast<std::int32_t>(prompt.size()))) != 0) {
        throw std::runtime_error("prompt evaluation failed");
    }

    Utf8Assembler utf8;
    for (int step = 0; step < request.max_new_tokens; ++step) {
        Token token = llama_sampler_sample(sampler.get(), ctx, -1);
        if (llama_vocab_is_eog(vocab_, token)) {
            result.reason = StopReason::eos;
            break;
        }
        ++result.completion_tokens;

        const std::string text = utf8.push(token_to_piece(token));
        if (!text.empty() && !on_piece(text)) {
            result.reason = StopReason::cancelled;
            return result;
        }

        if (step + 1 == request.max_new_tokens) {
            break;  // no need to evaluate a token whose successor we won't sample
        }
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            throw std::runtime_error("token evaluation failed");
        }
    }

    if (const std::string tail = utf8.flush(); !tail.empty()) {
        on_piece(tail);
    }
    return result;
}

}  // namespace lisa
