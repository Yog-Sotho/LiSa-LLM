#pragma once

#include "engine.hpp"
#include "request.hpp"
#include "utils.hpp"

#include <exception>

// JSON encoding of the broker <-> worker messages. After init, every message carries the
// "id" of its request, so requests are multiplexed and generations run concurrently.
//
//   broker -> worker: {"op":"init","settings":{...}}            once, before the sandbox
//                     {"op":"tokenize","id":N,"text":"..."}
//                     {"op":"tokenize_chat","id":N,"messages":[...]}
//                     {"op":"generate","id":N,"tokens":[...],"params":{...}}
//                     {"op":"cancel","id":N}                       stop generation N
//                     {"op":"probe","id":N,"name":"..."}           test builds only
//   worker -> broker: {"ready":{model info}} | {"fatal":"..."}     reply to init
//                     {"id":N,"tokens":[...]}
//                     {"id":N,"piece":"..."}* then {"id":N,"done":{result}}
//                     {"id":N,"error":{"kind":"request"|"internal","message":"...",...}}
namespace lisa {

void to_json(json& j, const SamplingParams& s);
void from_json(const json& j, SamplingParams& s);
void to_json(json& j, const GenerationParams& p);
void from_json(const json& j, GenerationParams& p);
void to_json(json& j, const ChatMessage& m);
void from_json(const json& j, ChatMessage& m);
void to_json(json& j, const ModelInfo& info);
void from_json(const json& j, ModelInfo& info);
void to_json(json& j, const GenerationResult& r);
void from_json(const json& j, GenerationResult& r);

// The part of Config the worker needs. Secrets (api_key, TLS keys) never cross over.
struct WorkerSettings {
    std::string chat_template;
    int n_threads = 4;
    std::uint32_t max_context = 2048;
    int parallel_slots = 1;
    std::string flash_attention = "auto";
    std::string kv_cache_type = "f16";
    std::uint64_t memory_limit_bytes = 0;
    std::string cgroup;  // delegated cgroup v2 parent directory, or empty
    bool seccomp_log = false;

    [[nodiscard]] static WorkerSettings from_config(const Config& cfg);
};
void to_json(json& j, const WorkerSettings& s);
void from_json(const json& j, WorkerSettings& s);

// {"error":{...}} for an exception thrown while serving a request.
[[nodiscard]] json error_reply(const std::exception& e);

// Rethrows an {"error":{...}} reply as RequestError or std::runtime_error.
[[noreturn]] void throw_error_reply(const json& reply);

}  // namespace lisa
