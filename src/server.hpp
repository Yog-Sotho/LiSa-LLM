#pragma once

#include "inference.hpp"
#include "utils.hpp"
#include <memory>

class HttpServer {
public:
    HttpServer(InferenceEngine& engine, const Config& cfg);
    void start();
    void stop();
    
private:
    InferenceEngine& engine_;
    Config cfg_;
    bool running_ = false;
};
