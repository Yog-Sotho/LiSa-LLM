#include "protocol.hpp"

#include <stdexcept>
#include <string>

namespace lisa {

void to_json(json& j, const SamplingParams& s) {
    j = json{{"temperature", s.temperature},
             {"top_p", s.top_p},
             {"top_k", s.top_k},
             {"min_p", s.min_p},
             {"presence_penalty", s.presence_penalty},
             {"frequency_penalty", s.frequency_penalty},
             {"repeat_penalty", s.repeat_penalty},
             {"seed", s.seed}};
}

void from_json(const json& j, SamplingParams& s) {
    j.at("temperature").get_to(s.temperature);
    j.at("top_p").get_to(s.top_p);
    j.at("top_k").get_to(s.top_k);
    j.at("min_p").get_to(s.min_p);
    j.at("presence_penalty").get_to(s.presence_penalty);
    j.at("frequency_penalty").get_to(s.frequency_penalty);
    j.at("repeat_penalty").get_to(s.repeat_penalty);
    j.at("seed").get_to(s.seed);
}

void to_json(json& j, const GenerationParams& p) {
    j = json{{"max_tokens", p.max_tokens},
             {"max_tokens_explicit", p.max_tokens_explicit},
             {"stop", p.stop},
             {"sampling", p.sampling},
             {"stream", p.stream},
             {"include_usage", p.include_usage},
             {"json_object", p.json_object}};
}

void from_json(const json& j, GenerationParams& p) {
    j.at("max_tokens").get_to(p.max_tokens);
    j.at("max_tokens_explicit").get_to(p.max_tokens_explicit);
    j.at("stop").get_to(p.stop);
    j.at("sampling").get_to(p.sampling);
    j.at("stream").get_to(p.stream);
    j.at("include_usage").get_to(p.include_usage);
    j.at("json_object").get_to(p.json_object);
}

void to_json(json& j, const ChatMessage& m) {
    j = json{{"role", m.role}, {"content", m.content}};
}

void from_json(const json& j, ChatMessage& m) {
    j.at("role").get_to(m.role);
    j.at("content").get_to(m.content);
}

void to_json(json& j, const ModelInfo& info) {
    j = json{{"description", info.description},
             {"chat_template", info.chat_template},
             {"n_layer", info.n_layer},
             {"n_ctx_train", info.n_ctx_train},
             {"n_ctx", info.n_ctx},
             {"n_params", info.n_params},
             {"gpu_offload", info.gpu_offload}};
}

void from_json(const json& j, ModelInfo& info) {
    j.at("description").get_to(info.description);
    j.at("chat_template").get_to(info.chat_template);
    j.at("n_layer").get_to(info.n_layer);
    j.at("n_ctx_train").get_to(info.n_ctx_train);
    j.at("n_ctx").get_to(info.n_ctx);
    j.at("n_params").get_to(info.n_params);
    j.at("gpu_offload").get_to(info.gpu_offload);
}

namespace {

const char* reason_name(StopReason reason) {
    switch (reason) {
        case StopReason::eos:
            return "eos";
        case StopReason::stop_sequence:
            return "stop_sequence";
        case StopReason::length:
            return "length";
        case StopReason::cancelled:
            break;
    }
    return "cancelled";
}

StopReason reason_from_name(const std::string& name) {
    if (name == "eos") {
        return StopReason::eos;
    }
    if (name == "stop_sequence") {
        return StopReason::stop_sequence;
    }
    if (name == "length") {
        return StopReason::length;
    }
    if (name == "cancelled") {
        return StopReason::cancelled;
    }
    throw std::runtime_error("unknown stop reason '" + name + "'");
}

}  // namespace

void to_json(json& j, const GenerationResult& r) {
    j = json{{"prompt_tokens", r.prompt_tokens},
             {"cached_tokens", r.cached_tokens},
             {"completion_tokens", r.completion_tokens},
             {"reason", reason_name(r.reason)}};
}

void from_json(const json& j, GenerationResult& r) {
    j.at("prompt_tokens").get_to(r.prompt_tokens);
    j.at("cached_tokens").get_to(r.cached_tokens);
    j.at("completion_tokens").get_to(r.completion_tokens);
    r.reason = reason_from_name(j.at("reason").get<std::string>());
}

WorkerSettings WorkerSettings::from_config(const Config& cfg) {
    WorkerSettings s;
    s.chat_template = cfg.chat_template;
    s.n_threads = cfg.n_threads;
    s.max_context = cfg.max_context;
    s.parallel_slots = cfg.parallel_slots;
    s.flash_attention = cfg.flash_attention;
    s.kv_cache_type = cfg.kv_cache_type;
    s.memory_limit_bytes = cfg.memory_limit_bytes;
    s.cgroup = cfg.sandbox_cgroup;
    s.seccomp_log = cfg.sandbox_seccomp_log;
    return s;
}

void to_json(json& j, const WorkerSettings& s) {
    j = json{{"chat_template", s.chat_template},
             {"n_threads", s.n_threads},
             {"max_context", s.max_context},
             {"parallel_slots", s.parallel_slots},
             {"flash_attention", s.flash_attention},
             {"kv_cache_type", s.kv_cache_type},
             {"memory_limit_bytes", s.memory_limit_bytes},
             {"cgroup", s.cgroup},
             {"seccomp_log", s.seccomp_log}};
}

void from_json(const json& j, WorkerSettings& s) {
    j.at("chat_template").get_to(s.chat_template);
    j.at("n_threads").get_to(s.n_threads);
    j.at("max_context").get_to(s.max_context);
    j.at("parallel_slots").get_to(s.parallel_slots);
    j.at("flash_attention").get_to(s.flash_attention);
    j.at("kv_cache_type").get_to(s.kv_cache_type);
    j.at("memory_limit_bytes").get_to(s.memory_limit_bytes);
    j.at("cgroup").get_to(s.cgroup);
    j.at("seccomp_log").get_to(s.seccomp_log);
}

json error_reply(const std::exception& e) {
    if (const auto* request = dynamic_cast<const RequestError*>(&e)) {
        return json{
            {"error",
             {{"kind", "request"}, {"message", request->what()}, {"param", request->param()}}}};
    }
    return json{{"error", {{"kind", "internal"}, {"message", e.what()}}}};
}

void throw_error_reply(const json& reply) {
    const json& error = reply.at("error");
    const std::string message = error.value("message", "worker error");
    if (error.value("kind", "") == "request") {
        throw RequestError(message, error.value("param", ""));
    }
    throw std::runtime_error(message);
}

}  // namespace lisa
