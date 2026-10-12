cmake_minimum_required(VERSION 3.28)
project(LiSa-LLM VERSION 1.0.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(LISA_TLS "Enable HTTPS support (requires OpenSSL)" ON)
option(LISA_SANDBOX "Build the Linux sandbox (requires libseccomp)" ON)
option(LISA_CUDA "Build the inference engine with CUDA offload" OFF)
option(LISA_NATIVE "Optimise the engine for the build machine's CPU (disable for portable binaries)" ON)
option(LISA_HARDENING "Compile with stack protector, FORTIFY_SOURCE, RELRO/now and PIE" ON)
option(LISA_WARNINGS_AS_ERRORS "Treat compiler warnings in first-party code as errors" OFF)
option(LISA_BUILD_TESTS "Build the unit tests" ${PROJECT_IS_TOP_LEVEL})
set(LISA_SANITIZERS "" CACHE STRING "Comma-separated sanitizers for first-party code, e.g. address,undefined")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# No C++20 modules here. Scanning would add @*.modmap response files to
# compile_commands.json that only exist after a build, which breaks clang-tidy.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(LISA_SANDBOX AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "LISA_SANDBOX is only supported on Linux; disabling it")
    set(LISA_SANDBOX OFF)
endif()

include(CompilerSettings)
include(Dependencies)

# -----------------------------------------------------------------------------
# First-party code
# -----------------------------------------------------------------------------
configure_file(src/version.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/generated/lisa/version.hpp" @ONLY)

add_library(lisa_core STATIC
    src/utils.cpp
    src/request.cpp
    src/stop.cpp
    src/openai.cpp
    src/utf8.cpp
    src/inference.cpp
    src/server.cpp
    src/sandbox.cpp
    src/ipc.cpp
    src/protocol.cpp
    src/model_file.cpp
    src/worker.cpp
    src/sandboxed_engine.cpp
)
target_include_directories(lisa_core PUBLIC
    src
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_link_libraries(lisa_core
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        llama
        httplib::httplib
        yaml-cpp::yaml-cpp
        OpenSSL::Crypto
        Threads::Threads
)
lisa_configure_target(lisa_core)
if(LISA_CUDA)
    target_compile_definitions(lisa_core PRIVATE LISA_CUDA_BUILD)
endif()

if(LISA_SANDBOX)
    target_link_libraries(lisa_core PRIVATE PkgConfig::LIBSECCOMP)
    target_compile_definitions(lisa_core PRIVATE LISA_ENABLE_SANDBOX)
endif()

add_executable(lisa src/main.cpp)
target_link_libraries(lisa PRIVATE lisa_core)
lisa_configure_target(lisa)

install(TARGETS lisa RUNTIME DESTINATION bin)

if(LISA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "LiSa-LLM ${PROJECT_VERSION}: TLS=${LISA_TLS} SANDBOX=${LISA_SANDBOX} CUDA=${LISA_CUDA} "
               "NATIVE=${LISA_NATIVE} SANITIZERS='${LISA_SANITIZERS}' TESTS=${LISA_BUILD_TESTS}")
