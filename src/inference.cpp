#include "inference.hpp"

#include "stop.hpp"
#include "utf8.hpp"

#include <llama.h>
#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <stdexcept>

namespace lisa {

namespace {

constexpr std::size_t kPieceBufferSize = 64;
constexpr std::int32_t kPenaltyLastN = 64;  // llama.cpp's default repetition window
constexpr std::size_t kMaxBuiltinTemplates = 256;
constexpr const char* kFallbackChatTemplate = "chatml";

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

// Same order as llama.cpp's defaults: penalties → top-k → top-p → min-p → temperature →
// random draw (or greedy when temperature is 0).
SamplerPtr make_sampler(const SamplingParams& s, std::int32_t n_vocab) {
    SamplerPtr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!chain) {
        throw std::runtime_error("failed to create sampler");
    }
    llama_sampler* c = chain.get();
    if (s.repeat_penalty != 1.0f || s.frequency_penalty != 0.0f || s.presence_penalty != 0.0f) {
        llama_sampler_chain_add(
            c, llama_sampler_init_penalties(n_vocab, kPenaltyLastN, s.repeat_penalty,
                                            s.frequency_penalty, s.presence_penalty));
    }
    if (s.temperature <= 0.0f) {
        llama_sampler_chain_add(c, llama_sampler_init_greedy());
        return chain;
    }
    if (s.top_k > 0) {
        llama_sampler_chain_add(c, llama_sampler_init_top_k(s.top_k));
    }
    if (s.top_p < 1.0f) {
        llama_sampler_chain_add(c, llama_sampler_init_top_p(s.top_p, 1));
    }
    if (s.min_p > 0.0f) {
        llama_sampler_chain_add(c, llama_sampler_init_min_p(s.min_p, 1));
    }
    llama_sampler_chain_add(c, llama_sampler_init_temp(s.temperature));
    llama_sampler_chain_add(c, llama_sampler_init_dist(s.seed));
    return chain;
}

std::vector<std::string> builtin_chat_templates() {
    std::array<const char*, kMaxBuiltinTemplates> names{};
    const std::int32_t n = llama_chat_builtin_templates(names.data(), names.size());
    std::vector<std::string> result;
    for (std::int32_t i = 0; i < n && static_cast<std::size_t>(i) < names.size(); ++i) {
        result.emplace_back(names[static_cast<std::size_t>(i)]);
    }
    return result;
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

    select_chat_template(cfg.chat_template);
}

InferenceEngine::~InferenceEngine() = default;

void InferenceEngine::select_chat_template(const std::string& configured) {
    const std::vector<ChatMessage> probe = {{"system", "s"}, {"user", "u"}};

    if (!configured.empty()) {
        const auto names = builtin_chat_templates();
        if (std::find(names.begin(), names.end(), configured) == names.end()) {
            std::string list;
            for (const auto& name : names) {
                list += (list.empty() ? "" : ", ") + name;
            }
            throw std::runtime_error("unknown chat_template '" + configured +
                                     "'; built-in templates: " + list);
        }
        chat_template_ = configured;
        info_.chat_template = configured + " (configured)";
        return;
    }

    if (const char* model_template = llama_model_chat_template(model_.get(), nullptr)) {
        chat_template_ = model_template;
        try {
            (void)apply_chat_template(probe);
            info_.chat_template = "model (auto-detected)";
            return;
        } catch (const RequestError&) {
            log_warn(
                "The model's chat template is not one of the built-in formats; using ChatML. "
                "Set chat_template in the config to choose another format.");
        }
    } else {
        log_warn(
            "The model has no chat template; using ChatML for /v1/chat/completions. "
            "Set chat_template in the config to choose another format.");
    }
    chat_template_ = kFallbackChatTemplate;
    info_.chat_template = std::string(kFallbackChatTemplate) + " (fallback)";
}

std::vector<Token> InferenceEngine::tokenize_text(std::string_view text, bool add_special,
                                                  bool parse_special) const {
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw RequestError("prompt is too large", "");
    }
    const auto text_len = static_cast<std::int32_t>(text.size());

    // Usually enough (≤ 1 token per byte + BOS/EOS); retried with the exact size otherwise.
    std::vector<Token> tokens(text.size() + 2);
    std::int32_t n =
        llama_tokenize(vocab_, text.data(), text_len, tokens.data(),
                       static_cast<std::int32_t>(tokens.size()), add_special, parse_special);
    if (n == INT32_MIN) {
        throw RequestError("prompt is too large", "");
    }
    if (n < 0) {
        tokens.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(vocab_, text.data(), text_len, tokens.data(),
                           static_cast<std::int32_t>(tokens.size()), add_special, parse_special);
        if (n < 0) {
            throw RequestError("prompt tokenization failed", "");
        }
    }
    tokens.resize(static_cast<std::size_t>(n));
    return tokens;
}

std::vector<Token> InferenceEngine::tokenize(std::string_view text) const {
    std::vector<Token> tokens = tokenize_text(text, /*add_special=*/true, /*parse_special=*/false);
    if (tokens.empty()) {
        throw RequestError("prompt produced no tokens", "prompt");
    }
    return tokens;
}

std::string InferenceEngine::apply_chat_template(const std::vector<ChatMessage>& messages) const {
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    std::size_t total = 0;
    for (const ChatMessage& m : messages) {
        chat.push_back({m.role.c_str(), m.content.c_str()});
        total += m.role.size() + m.content.size();
    }
    if (total > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / 4)) {
        throw RequestError("messages are too large", "messages");
    }

    std::string formatted(2 * total + 256, '\0');
    std::int32_t n = llama_chat_apply_template(chat_template_.c_str(), chat.data(), chat.size(),
                                               /*add_ass=*/true, formatted.data(),
                                               static_cast<std::int32_t>(formatted.size()));
    if (n > static_cast<std::int32_t>(formatted.size())) {
        formatted.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(chat_template_.c_str(), chat.data(), chat.size(), true,
                                      formatted.data(), n);
    }
    if (n < 0) {
        throw RequestError("the chat template cannot format these messages", "messages");
    }
    formatted.resize(static_cast<std::size_t>(n));
    return formatted;
}

std::vector<Token> InferenceEngine::tokenize_chat(const std::vector<ChatMessage>& messages) const {
    for (std::size_t i = 0; i < messages.size(); ++i) {
        // parse_special only changes the result when the text spells a special token.
        const std::string& content = messages[i].content;
        if (tokenize_text(content, false, true) != tokenize_text(content, false, false)) {
            throw RequestError("message content must not contain special-token text",
                               "messages[" + std::to_string(i) + "].content");
        }
    }
    // The template's own markers (e.g. <|im_start|>) must become special tokens.
    std::vector<Token> tokens =
        tokenize_text(apply_chat_template(messages), /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) {
        throw RequestError("messages produced no tokens", "messages");
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
                                           const GenerationParams& params,
                                           const PieceCallback& on_piece) {
    if (prompt_tokens.empty() || params.max_tokens < 1) {
        throw RequestError("empty prompt or max_tokens < 1", "max_tokens");
    }
    if (prompt_tokens.size() + static_cast<std::size_t>(params.max_tokens) > info_.n_ctx) {
        throw RequestError("prompt (" + std::to_string(prompt_tokens.size()) +
                               " tokens) plus max_tokens exceeds the context window of " +
                               std::to_string(info_.n_ctx) + " tokens",
                           "max_tokens");
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    llama_context* ctx = ctx_.get();
    llama_memory_clear(llama_get_memory(ctx), true);
    const SamplerPtr sampler = make_sampler(params.sampling, llama_vocab_n_tokens(vocab_));

    GenerationResult result;
    result.prompt_tokens = prompt_tokens.size();

    // llama_batch takes a non-const pointer; it does not modify the tokens.
    std::vector<Token> prompt = prompt_tokens;
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(),
                                              static_cast<std::int32_t>(prompt.size()))) != 0) {
        throw std::runtime_error("prompt evaluation failed");
    }

    Utf8Assembler utf8;
    StopMatcher stop(params.stop);
    for (int step = 0; step < params.max_tokens; ++step) {
        Token token = llama_sampler_sample(sampler.get(), ctx, -1);
        if (llama_vocab_is_eog(vocab_, token)) {
            result.reason = StopReason::eos;
            break;
        }
        ++result.completion_tokens;

        const std::string text = stop.push(utf8.push(token_to_piece(token)));
        if (!text.empty() && !on_piece(text)) {
            result.reason = StopReason::cancelled;
            return result;
        }
        if (stop.stopped()) {
            result.reason = StopReason::stop_sequence;
            return result;
        }

        if (step + 1 == params.max_tokens) {
            break;  // no need to evaluate a token whose successor we won't sample
        }
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            throw std::runtime_error("token evaluation failed");
        }
    }

    // Generation ended (EOS or length): release anything held back for UTF-8 or stops.
    std::string tail = stop.push(utf8.flush());
    if (stop.stopped()) {
        result.reason = StopReason::stop_sequence;
    } else {
        tail += stop.flush();
    }
    if (!tail.empty()) {
        on_piece(tail);
    }
    return result;
}

}  // namespace lisa
