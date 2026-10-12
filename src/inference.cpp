#include "inference.hpp"

#include "stop.hpp"
#include "utf8.hpp"

#include <llama.h>
#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lisa {

namespace {

constexpr std::size_t kPieceBufferSize = 64;
constexpr std::int32_t kPenaltyLastN = 64;  // llama.cpp's default repetition window
constexpr std::size_t kMaxBuiltinTemplates = 256;
constexpr const char* kFallbackChatTemplate = "chatml";
constexpr std::uint32_t kBatchTokens = 512;  // tokens evaluated per scheduler step
constexpr std::uint32_t kKvCellAlignment = 256;
// Copying a whole slot cache only pays off when it saves evaluating this many tokens.
constexpr std::size_t kMinCopiedTokens = 32;

ggml_type kv_type(const std::string& name) {
    if (name == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (name == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    return GGML_TYPE_F16;
}

llama_flash_attn_type flash_attn_type(const std::string& name) {
    if (name == "on") {
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (name == "off") {
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}

std::size_t common_prefix(const std::vector<Token>& a, const std::vector<Token>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

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

// JSON object grammar: grammars/json.gbnf from llama.cpp (MIT licence).
constexpr const char* kJsonGrammar = R"gbnf(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
)gbnf";

// A grammar (JSON mode) first, so later steps only see allowed tokens; then llama.cpp's
// default order: penalties → top-k → top-p → min-p → temperature → random draw (or
// greedy when temperature is 0).
SamplerPtr make_sampler(const GenerationParams& params, const llama_vocab* vocab) {
    const SamplingParams& s = params.sampling;
    const std::int32_t n_vocab = llama_vocab_n_tokens(vocab);
    SamplerPtr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!chain) {
        throw std::runtime_error("failed to create sampler");
    }
    llama_sampler* c = chain.get();
    if (params.json_object) {
        llama_sampler* grammar = llama_sampler_init_grammar(vocab, kJsonGrammar, "root");
        if (grammar == nullptr) {
            throw std::runtime_error("failed to create the JSON grammar sampler");
        }
        llama_sampler_chain_add(c, grammar);
    }
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

// A llama_batch with room for `capacity` tokens of one sequence each, reused every step.
class InferenceEngine::Batch {
public:
    explicit Batch(std::int32_t capacity) : batch_(llama_batch_init(capacity, 0, 1)) {}
    ~Batch() { llama_batch_free(batch_); }
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&) = delete;
    Batch& operator=(Batch&&) = delete;

    [[nodiscard]] llama_batch& get() noexcept { return batch_; }

private:
    llama_batch batch_;
};

// One generation request.
struct InferenceEngine::Job {
    RequestId id = 0;
    std::vector<Token> prompt;
    GenerationParams params;
    Callbacks callbacks;
    std::atomic<bool> cancelled{false};
};

// One sequence of the batch. Its KV cache keeps the tokens in `cache` (positions
// 0..cache.size()-1) after a request ends, for prefix reuse by the next one.
struct InferenceEngine::Slot {
    llama_seq_id seq = 0;
    std::shared_ptr<Job> job;  // null when free
    std::vector<Token> cache;  // tokens evaluated into this sequence's KV cache
    std::size_t prefill = 0;   // next prompt token to evaluate
    Token next = 0;            // sampled token to evaluate in the next step
    std::int32_t output = -1;  // batch index whose logits this slot samples from
    SamplerPtr sampler;
    Utf8Assembler utf8;
    StopMatcher stop{std::vector<std::string>{}};
    GenerationResult result;
    std::uint64_t last_used = 0;

    [[nodiscard]] bool prefilling() const { return job && prefill < job->prompt.size(); }
};

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

    const auto slot_count = static_cast<std::uint32_t>(std::max(1, cfg.parallel_slots));
    llama_context_params ctx_params = llama_context_default_params();
    // Every slot gets a full max_context of KV cache, so an admitted request always fits.
    // llama.cpp rounds each sequence's share up to a multiple of 256; doing it here keeps
    // the total divisible by the slot count.
    const std::uint32_t slot_cells =
        (n_ctx + kKvCellAlignment - 1) / kKvCellAlignment * kKvCellAlignment;
    ctx_params.n_ctx = slot_cells * slot_count;
    ctx_params.n_seq_max = slot_count;
    ctx_params.kv_unified = false;  // separate per-slot caches: faster without shared prompts
    n_batch_ = std::max(kBatchTokens, slot_count);
    ctx_params.n_batch = n_batch_;
    ctx_params.n_ubatch = n_batch_;
    const int threads = cfg.n_threads > 0 ? cfg.n_threads : available_cpus();
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;
    ctx_params.flash_attn_type = flash_attn_type(cfg.flash_attention);
    ctx_params.type_k = kv_type(cfg.kv_cache_type);
    ctx_params.type_v = kv_type(cfg.kv_cache_type);
    ctx_params.no_perf = true;

    ctx_.reset(llama_init_from_model(model_.get(), ctx_params));
    if (!ctx_) {
        throw std::runtime_error(
            "failed to create inference context (if memory ran out, lower parallel_slots or "
            "max_context, or use kv_cache_type q8_0)");
    }

    std::string description(256, '\0');
    const std::int32_t len = llama_model_desc(model_.get(), description.data(), description.size());
    description.resize(len > 0 ? std::min(static_cast<std::size_t>(len), description.size()) : 0);

    info_.description = std::move(description);
    info_.n_layer = llama_model_n_layer(model_.get());
    info_.n_ctx_train = n_ctx_train;
    // llama.cpp may pad the allocation; requests are still limited to what was configured.
    info_.n_ctx = std::min(n_ctx, llama_n_ctx_seq(ctx_.get()));
    info_.n_params = llama_model_n_params(model_.get());
    // Recurrent and sliding-window caches can't be trimmed to an arbitrary prefix after
    // a copy, so they only reuse a slot's own cache.
    can_copy_sequences_ = !llama_model_is_recurrent(model_.get()) &&
                          !llama_model_is_hybrid(model_.get()) &&
                          llama_model_n_swa(model_.get()) == 0;
    info_.gpu_offload = llama_supports_gpu_offload() && model_params.n_gpu_layers != 0;

    select_chat_template(cfg.chat_template);

    batch_ = std::make_unique<Batch>(static_cast<std::int32_t>(n_batch_));
    for (std::uint32_t i = 0; i < slot_count; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->seq = static_cast<llama_seq_id>(i);
        slots_.push_back(std::move(slot));
    }
    scheduler_ = std::thread([this] { run_scheduler(); });
}

InferenceEngine::~InferenceEngine() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    scheduler_.join();
}

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

std::vector<Token> InferenceEngine::tokenize(std::string_view text) {
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

std::vector<Token> InferenceEngine::tokenize_chat(const std::vector<ChatMessage>& messages) {
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

InferenceEngine::RequestId InferenceEngine::submit(std::vector<Token> prompt_tokens,
                                                   GenerationParams params, Callbacks callbacks) {
    if (prompt_tokens.empty() || params.max_tokens < 1) {
        throw RequestError("empty prompt or max_tokens < 1", "max_tokens");
    }
    if (prompt_tokens.size() + static_cast<std::size_t>(params.max_tokens) > info_.n_ctx) {
        throw RequestError("prompt (" + std::to_string(prompt_tokens.size()) +
                               " tokens) plus max_tokens exceeds the context window of " +
                               std::to_string(info_.n_ctx) + " tokens",
                           "max_tokens");
    }
    auto job = std::make_shared<Job>();
    job->prompt = std::move(prompt_tokens);
    job->params = std::move(params);
    job->callbacks = std::move(callbacks);
    RequestId id = 0;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            throw std::runtime_error("the inference engine is shutting down");
        }
        id = next_id_++;
        job->id = id;
        jobs_.emplace(id, job);
        pending_.push_back(std::move(job));
    }
    wake_.notify_all();
    return id;
}

void InferenceEngine::cancel(RequestId id) {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto it = jobs_.find(id);
        if (it == jobs_.end()) {
            return;
        }
        it->second->cancelled = true;
    }
    wake_.notify_all();
}

GenerationResult InferenceEngine::generate(const std::vector<Token>& prompt_tokens,
                                           const GenerationParams& params,
                                           const PieceCallback& on_piece) {
    // The scheduler hands pieces over through this queue; `on_piece` runs here, so a slow
    // consumer never holds up the other requests in the batch.
    struct Pipe {
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<std::string> pieces;
        bool done = false;
        bool abandoned = false;  // the consumer stopped reading
        GenerationResult result;
        std::exception_ptr error;
    };
    auto pipe = std::make_shared<Pipe>();
    Callbacks callbacks;
    callbacks.on_piece = [pipe](std::string_view piece) {
        {
            const std::lock_guard<std::mutex> lock(pipe->mutex);
            if (pipe->abandoned) {
                return false;
            }
            pipe->pieces.emplace_back(piece);
        }
        pipe->changed.notify_one();
        return true;
    };
    callbacks.on_done = [pipe](const GenerationResult& result) {
        {
            const std::lock_guard<std::mutex> lock(pipe->mutex);
            pipe->result = result;
            pipe->done = true;
        }
        pipe->changed.notify_one();
    };
    callbacks.on_error = [pipe](std::exception_ptr error) {
        {
            const std::lock_guard<std::mutex> lock(pipe->mutex);
            pipe->error = std::move(error);
            pipe->done = true;
        }
        pipe->changed.notify_one();
    };
    const RequestId id = submit(prompt_tokens, params, std::move(callbacks));

    std::unique_lock<std::mutex> lock(pipe->mutex);
    while (true) {
        pipe->changed.wait(lock, [&] { return pipe->done || !pipe->pieces.empty(); });
        while (!pipe->pieces.empty()) {
            const std::string piece = std::move(pipe->pieces.front());
            pipe->pieces.pop_front();
            if (pipe->abandoned) {
                continue;
            }
            lock.unlock();
            bool keep_going = false;
            try {
                keep_going = on_piece(piece);
            } catch (...) {
                cancel(id);
                throw;
            }
            lock.lock();
            if (!keep_going) {
                pipe->abandoned = true;
                lock.unlock();
                cancel(id);
                lock.lock();
            }
        }
        if (pipe->done) {
            break;
        }
    }
    if (pipe->error) {
        std::rethrow_exception(pipe->error);
    }
    GenerationResult result = pipe->result;
    if (pipe->abandoned) {
        result.reason = StopReason::cancelled;
    }
    return result;
}

void InferenceEngine::run_scheduler() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        const auto busy = [this] {
            return std::any_of(slots_.begin(), slots_.end(),
                               [](const auto& slot) { return slot->job != nullptr; });
        };
        wake_.wait(lock, [&] { return stopping_ || !pending_.empty() || busy(); });
        if (stopping_) {
            break;
        }
        const std::vector<std::shared_ptr<Job>> cancelled = admit_pending_locked();
        lock.unlock();
        for (const auto& job : cancelled) {
            GenerationResult result;
            result.prompt_tokens = job->prompt.size();
            result.reason = StopReason::cancelled;
            job->callbacks.on_done(result);
        }
        try {
            step();
        } catch (...) {
            fail_active(std::current_exception());
        }
        lock.lock();
    }

    // Shutting down: nothing will run any more.
    std::vector<std::shared_ptr<Job>> left(pending_.begin(), pending_.end());
    pending_.clear();
    jobs_.clear();
    lock.unlock();
    for (const auto& slot : slots_) {
        if (slot->job) {
            left.push_back(std::move(slot->job));
        }
    }
    const auto error =
        std::make_exception_ptr(std::runtime_error("the inference engine is shutting down"));
    for (const auto& job : left) {
        job->callbacks.on_error(error);
    }
}

std::vector<std::shared_ptr<InferenceEngine::Job>> InferenceEngine::admit_pending_locked() {
    std::vector<std::shared_ptr<Job>> cancelled;
    while (!pending_.empty()) {
        std::shared_ptr<Job>& job = pending_.front();
        if (job->cancelled) {  // cancelled while waiting: finish without running
            jobs_.erase(job->id);
            cancelled.push_back(std::move(job));
            pending_.pop_front();
            continue;
        }
        // The free slot whose cache shares the longest prefix with the prompt; among
        // equals the least recently used, so recent caches survive longer.
        Slot* best = nullptr;
        std::size_t best_prefix = 0;
        for (const auto& slot : slots_) {
            if (slot->job) {
                continue;
            }
            const std::size_t prefix = common_prefix(slot->cache, job->prompt);
            if (best == nullptr || prefix > best_prefix ||
                (prefix == best_prefix && slot->last_used < best->last_used)) {
                best = slot.get();
                best_prefix = prefix;
            }
        }
        if (best == nullptr) {
            break;  // all slots busy
        }
        std::shared_ptr<Job> next = std::move(job);
        pending_.pop_front();
        start_job(*best, std::move(next));
    }
    return cancelled;
}

void InferenceEngine::start_job(Slot& slot, std::shared_ptr<Job> job) {
    llama_memory_t memory = llama_get_memory(ctx_.get());
    std::size_t own = common_prefix(slot.cache, job->prompt);

    // Another slot (busy or not) may hold much more of this prompt, e.g. a system prompt
    // shared by concurrent requests. Copying its cache beats evaluating the tokens again.
    const Slot* donor = nullptr;
    std::size_t donor_prefix = 0;
    for (const auto& other : slots_) {
        if (other.get() == &slot) {
            continue;
        }
        const std::size_t prefix = common_prefix(other->cache, job->prompt);
        if (prefix > donor_prefix) {
            donor = other.get();
            donor_prefix = prefix;
        }
    }
    if (donor != nullptr && can_copy_sequences_ && donor_prefix >= own + kMinCopiedTokens) {
        // Separate per-slot caches can only be copied whole; the extra part is trimmed
        // below. The data itself moves at the start of the next llama_decode.
        llama_memory_seq_cp(memory, donor->seq, slot.seq, -1, -1);
        slot.cache = donor->cache;
        own = donor_prefix;
    }

    // Keep the shared prefix; at least the last prompt token is evaluated again, because
    // its logits are needed to sample the first new token.
    std::size_t reuse = std::min(own, job->prompt.size() - 1);
    if (!llama_memory_seq_rm(memory, slot.seq, static_cast<llama_pos>(reuse), -1)) {
        // Some caches (e.g. recurrent models) cannot drop a suffix: start over.
        llama_memory_seq_rm(memory, slot.seq, -1, -1);
        reuse = 0;
    }
    slot.cache.resize(reuse);
    slot.prefill = reuse;
    slot.output = -1;
    slot.sampler = make_sampler(job->params, vocab_);
    slot.utf8 = Utf8Assembler();
    slot.stop = StopMatcher(job->params.stop);
    slot.result = GenerationResult();
    slot.result.prompt_tokens = job->prompt.size();
    slot.result.cached_tokens = reuse;
    slot.job = std::move(job);
}

void InferenceEngine::step() {
    // Finish cancelled requests first so they don't use the batch.
    for (const auto& slot : slots_) {
        if (slot->job && slot->job->cancelled) {
            finish(*slot, StopReason::cancelled, /*flush=*/false);
        }
    }

    llama_batch& batch = batch_->get();
    batch.n_tokens = 0;
    const auto add = [&](Slot& slot, Token token, bool wants_logits) {
        const std::int32_t i = batch.n_tokens++;
        batch.token[i] = token;
        batch.pos[i] = static_cast<llama_pos>(slot.cache.size());
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = slot.seq;
        batch.logits[i] = wants_logits ? 1 : 0;
        slot.cache.push_back(token);
        if (wants_logits) {
            slot.output = i;
        }
    };

    // One token for every generating request...
    for (const auto& slot : slots_) {
        slot->output = -1;
        if (slot->job && !slot->prefilling()) {
            add(*slot, slot->next, true);
        }
    }
    // ...then prompt chunks with the remaining room, oldest request first.
    for (const auto& slot : slots_) {
        if (!slot->job || !slot->prefilling()) {
            continue;
        }
        const std::vector<Token>& prompt = slot->job->prompt;
        while (slot->prefill < prompt.size() &&
               static_cast<std::uint32_t>(batch.n_tokens) < n_batch_) {
            const bool last = slot->prefill + 1 == prompt.size();
            add(*slot, prompt[slot->prefill], last);
            ++slot->prefill;
        }
    }
    if (batch.n_tokens == 0) {
        return;
    }

    if (llama_decode(ctx_.get(), batch) != 0) {
        throw std::runtime_error("token evaluation failed");
    }
    for (const auto& slot : slots_) {
        if (slot->job && slot->output >= 0) {
            accept_token(*slot, slot->output);
        }
    }
}

void InferenceEngine::accept_token(Slot& slot, std::int32_t batch_index) {
    const Token token = llama_sampler_sample(slot.sampler.get(), ctx_.get(), batch_index);
    if (llama_vocab_is_eog(vocab_, token)) {
        finish(slot, StopReason::eos, /*flush=*/true);
        return;
    }
    ++slot.result.completion_tokens;
    const std::string text = slot.stop.push(slot.utf8.push(token_to_piece(token)));
    if (!text.empty() && !slot.job->callbacks.on_piece(text)) {
        finish(slot, StopReason::cancelled, /*flush=*/false);
        return;
    }
    if (slot.stop.stopped()) {
        finish(slot, StopReason::stop_sequence, /*flush=*/false);
        return;
    }
    if (slot.result.completion_tokens >= static_cast<std::size_t>(slot.job->params.max_tokens)) {
        // No need to evaluate a token whose successor won't be sampled.
        finish(slot, StopReason::length, /*flush=*/true);
        return;
    }
    slot.next = token;
}

void InferenceEngine::finish(Slot& slot, StopReason reason, bool flush) {
    const std::shared_ptr<Job> job = std::move(slot.job);
    slot.result.reason = reason;
    if (flush) {
        // Release anything held back for UTF-8 completion or a possible stop sequence.
        std::string tail = slot.stop.push(slot.utf8.flush());
        if (slot.stop.stopped()) {
            slot.result.reason = StopReason::stop_sequence;
        } else {
            tail += slot.stop.flush();
        }
        if (!tail.empty()) {
            (void)job->callbacks.on_piece(tail);
        }
    }
    slot.sampler.reset();
    slot.last_used = ++use_counter_;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        jobs_.erase(job->id);
    }
    job->callbacks.on_done(slot.result);
}

void InferenceEngine::fail_active(const std::exception_ptr& error) {
    llama_memory_t memory = llama_get_memory(ctx_.get());
    for (const auto& slot : slots_) {
        if (!slot->job) {
            continue;
        }
        const std::shared_ptr<Job> job = std::move(slot->job);
        llama_memory_seq_rm(memory, slot->seq, -1, -1);
        slot->cache.clear();
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            jobs_.erase(job->id);
        }
        job->callbacks.on_error(error);
    }
}

void InferenceEngine::warm_up() {
    GenerationParams params;
    params.max_tokens = 1;
    params.sampling.temperature = 0.0f;
    (void)generate(tokenize("Hello"), params, [](std::string_view) { return true; });
}

}  // namespace lisa
