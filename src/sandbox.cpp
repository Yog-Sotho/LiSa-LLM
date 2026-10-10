#include "sandbox.hpp"

#ifdef LISA_ENABLE_SANDBOX

#include <sched.h>
#include <seccomp.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <string>

namespace lisa {

namespace {

constexpr uid_t kNobodyUid = 65534;
constexpr gid_t kNobodyGid = 65534;

void drop_privileges() {
    if (setgid(kNobodyGid) != 0 || setuid(kNobodyUid) != 0) {
        std::perror("drop_privileges");
        _exit(1);
    }
}

void install_seccomp() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL);
    if (ctx == nullptr) {
        std::perror("seccomp_init");
        _exit(1);
    }

    // Allow only strictly necessary syscalls
    const int allowed[] = {SCMP_SYS(read), SCMP_SYS(write), SCMP_SYS(close), SCMP_SYS(fstat),
                           SCMP_SYS(lseek), SCMP_SYS(mmap), SCMP_SYS(munmap), SCMP_SYS(brk),
                           SCMP_SYS(rt_sigaction), SCMP_SYS(rt_sigprocmask), SCMP_SYS(exit_group),
                           SCMP_SYS(getpid), SCMP_SYS(gettid), SCMP_SYS(getuid), SCMP_SYS(geteuid),
                           SCMP_SYS(getgid), SCMP_SYS(getegid), SCMP_SYS(arch_prctl),
                           // Network – for HTTP server
                           SCMP_SYS(socket), SCMP_SYS(bind), SCMP_SYS(listen), SCMP_SYS(accept4),
                           SCMP_SYS(recvfrom), SCMP_SYS(sendto), SCMP_SYS(epoll_create1),
                           SCMP_SYS(epoll_ctl), SCMP_SYS(epoll_wait), SCMP_SYS(getsockname),
                           SCMP_SYS(setsockopt), SCMP_SYS(getsockopt), SCMP_SYS(connect)};
    for (const int sys : allowed) {
        if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, sys, 0) < 0) {
            std::perror("seccomp_rule_add");
            seccomp_release(ctx);
            _exit(1);
        }
    }

    if (seccomp_load(ctx) < 0) {
        std::perror("seccomp_load");
        seccomp_release(ctx);
        _exit(1);
    }
    seccomp_release(ctx);
}

void setup_memory_limit(std::uint64_t limit_bytes) {
    // Create a new cgroup v2 subdirectory for this process
    const pid_t pid = getpid();
    const std::string cgroup_path = "/sys/fs/cgroup/lisa_" + std::to_string(pid);
    if (mkdir(cgroup_path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::perror("mkdir cgroup");
        return;
    }
    std::ofstream procs(cgroup_path + "/cgroup.procs");
    if (procs) {
        procs << pid << '\n';
        procs.close();
        std::ofstream max(cgroup_path + "/memory.max");
        if (max) {
            max << limit_bytes << '\n';
        }
    }
}

void pivot_root_sandbox() {
    // Create a temporary directory as new root
    char tmpdir[] = "/tmp/lisa_sandbox_XXXXXX";
    if (mkdtemp(tmpdir) == nullptr) {
        std::perror("mkdtemp");
        _exit(1);
    }
    // Move current root to old_root and pivot
    if (mount(tmpdir, tmpdir, "tmpfs", MS_NOSUID | MS_NODEV, nullptr) != 0) {
        std::perror("mount tmpfs");
        _exit(1);
    }
    if (chdir(tmpdir) != 0) {
        std::perror("chdir");
        _exit(1);
    }
    // glibc has no pivot_root() wrapper.
    if (syscall(SYS_pivot_root, ".", ".") != 0) {
        std::perror("pivot_root");
        _exit(1);
    }
    if (chroot(".") != 0) {
        std::perror("chroot");
        _exit(1);
    }
    if (umount2("/", MNT_DETACH) != 0) {
        // not fatal
    }
}

}  // namespace

bool sandbox_available() noexcept {
    return true;
}

int sandbox_init(const Config& cfg) {
    // Unshare namespaces
    if (unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET) != 0) {
        std::perror("unshare");
        return 1;
    }

    // Remount everything as private
    if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
        std::perror("mount private");
    }

    // Setup memory limit before pivot_root (cgroup path may be affected)
    if (cfg.memory_limit_bytes > 0) {
        setup_memory_limit(cfg.memory_limit_bytes);
    }

    // Isolate filesystem
    pivot_root_sandbox();

    // Drop capabilities and become nobody
    drop_privileges();

    // Apply seccomp filter
    install_seccomp();

    return 0;
}

}  // namespace lisa

#else  // !LISA_ENABLE_SANDBOX

namespace lisa {

bool sandbox_available() noexcept {
    return false;
}

int sandbox_init(const Config& /*cfg*/) {
    return 1;
}

}  // namespace lisa

#endif
