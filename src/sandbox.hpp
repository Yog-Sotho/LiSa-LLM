#pragma once

#include "utils.hpp"

// Setup the sandbox in the child process.
// Returns 0 on success, exits on failure.
int sandbox_init(const Config& cfg);
