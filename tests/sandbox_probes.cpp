// Escape attempts executed inside the sandboxed worker (test builds only). Each probe
// either reports what happened or gets the worker killed, which the test then expects.

#include "sandbox.hpp"
#include "utils.hpp"

#include <fcntl.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <string>
#include <thread>

namespace {

lisa::json errno_result(long rc) {
    return lisa::json{{"rc", rc}, {"errno", rc < 0 ? errno : 0}};
}

}  // namespace

lisa::json lisa_test_probe(const lisa::json& request) {
    const std::string name = request.value("name", "");
    const std::string path = request.value("path", "");

    if (name == "stat") {  // newfstatat is allowed: shows what the root contains
        struct stat st {};
        return errno_result(::stat(path.c_str(), &st));
    }
    if (name == "open") {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        const lisa::json result = errno_result(fd);
        if (fd >= 0) {
            ::close(fd);
        }
        return result;
    }
    if (name == "create") {
        const int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0600);
        const lisa::json result = errno_result(fd);
        if (fd >= 0) {
            ::close(fd);
        }
        return result;
    }
    if (name == "pid") {
        return lisa::json{{"pid", ::getpid()}};
    }
    if (name == "thread") {
        int value = 0;
        std::thread([&value] { value = 42; }).join();
        return lisa::json{{"value", value}};
    }
    if (name == "landlock") {
        return lisa::json{{"abi", lisa::landlock_abi_in_effect()}};
    }

    // Everything below must be blocked; the expected outcome is a dead worker.
    if (name == "socket") {
        return errno_result(::socket(AF_INET, SOCK_STREAM, 0));
    }
    if (name == "execve") {
        char sh[] = "/bin/sh";
        char* argv[] = {sh, nullptr};
        return errno_result(::execve(sh, argv, nullptr));
    }
    if (name == "fork") {  // clone without CLONE_THREAD
        const long rc = ::syscall(SYS_clone, SIGCHLD, 0, 0, 0, 0);
        if (rc == 0) {
            _exit(0);
        }
        return errno_result(rc);
    }
    if (name == "mount") {
        return errno_result(::mount("none", "/", "tmpfs", 0, nullptr));
    }
    if (name == "unshare") {
        return errno_result(::unshare(CLONE_NEWUSER));
    }
    if (name == "ptrace") {
        return errno_result(::ptrace(PTRACE_ATTACH, request.value("pid", 1), nullptr, nullptr));
    }
    if (name == "kill") {
        return errno_result(::kill(request.value("pid", 1), 0));
    }
    if (name == "setuid") {
        return errno_result(::setuid(0));
    }
    return lisa::json{{"unknown_probe", name}};
}
