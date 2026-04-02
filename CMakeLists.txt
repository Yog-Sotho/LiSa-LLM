cmake_minimum_required(VERSION 3.20)
project(LiSa-LLM VERSION 1.0.0 LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(STATIC_BUILD "Link everything statically" OFF)
option(ENABLE_TLS "Enable HTTPS support (requires OpenSSL)" ON)
option(ENABLE_SANDBOX "Enable sandbox isolation (Linux only)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(STATIC_BUILD)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_EXE_LINKER_FLAGS "-static -pthread")
endif()

# -----------------------------------------------------------------------------
# Dependencies (assume third_party contains the required libraries)
# -----------------------------------------------------------------------------
# GGML (static)
add_library(ggml STATIC
    third_party/ggml/ggml.c
    third_party/ggml/ggml-alloc.c
    third_party/ggml/ggml-nn.c
)
target_include_directories(ggml PUBLIC third_party/ggml)
target_compile_definitions(ggml PRIVATE GGML_USE_CPU)

# cpp-httplib (header only)
add_library(httplib INTERFACE)
target_include_directories(httplib INTERFACE third_party/cpp-httplib)

# nlohmann/json (header only)
add_library(json INTERFACE)
target_include_directories(json INTERFACE third_party/json/single_include)

# yaml-cpp (static or shared)
find_package(yaml-cpp REQUIRED)

# OpenSSL (for TLS and SHA256)
if(ENABLE_TLS)
    find_package(OpenSSL REQUIRED)
endif()

# seccomp (only on Linux)
if(ENABLE_SANDBOX AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBSECCOMP REQUIRED libseccomp)
    add_definitions(-DHAVE_SECCOMP)
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
set(SRCS
    src/main.cpp
    src/server.cpp
    src/inference.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/sandbox.cpp
    src/utils.cpp
)

add_executable(lisa ${SRCS})
target_include_directories(lisa PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_link_libraries(lisa
    PRIVATE
    ggml
    httplib
    json
    yaml-cpp
    pthread
    dl
    rt
)

if(ENABLE_TLS)
    target_link_libraries(lisa PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(lisa PRIVATE LISA_ENABLE_TLS)
endif()

if(ENABLE_SANDBOX AND LIBSECCOMP_FOUND)
    target_link_libraries(lisa PRIVATE ${LIBSECCOMP_LIBRARIES})
    target_compile_definitions(lisa PRIVATE LISA_ENABLE_SANDBOX)
    target_include_directories(lisa PRIVATE ${LIBSECCOMP_INCLUDE_DIRS})
endif()

# Install
install(TARGETS lisa RUNTIME DESTINATION bin)
