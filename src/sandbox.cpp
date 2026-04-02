#include "utils.hpp"
#include <unistd.h>
#include <sys/prctl.h>
#include <seccomp.h>
#include <sched.h>
#include <sys/mount.h>
#include <fstream>
#include <cstring>
#include <cerrno>

// ---------------------------------------------------------------------
// Drop privileges to an unprivileged user (nobody)
// ---------------------------------------------------------------------
static void drop_privileges() {
    if (setgid(65534) != 0 || setuid(65534) != 0) {
        perror("drop_privileges");
        _exit(1);
    }
}

// ---------------------------------------------------------------------
// Minimal seccomp filter – only the syscalls we actually need
// ---------------------------------------------------------------------
static void install_seccomp() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL);
    if (!ctx) { perror("seccomp_init"); _exit(1); }

    const int whitelist[] = {
        SCMP_SYS(read), SCMP_SYS(write), SCMP_SYS(open), SCMP_SYS(close),
        SCMP_SYS(fstat), SCMP_SYS(lseek), SCMP_SYS(mmap), SCMP_SYS(munmap),
        SCMP_SYS(brk), SCMP_SYS(rt_sigaction), SCMP_SYS(rt_sigprocmask),
        SCMP_SYS(ioctl), SCMP_SYS(poll), SCMP_SYS(select), SCMP_SYS(epoll_wait),
        SCMP_SYS(socket), SCMP_SYS(connect), SCMP_SYS(sendto), SCMP_SYS(recvfrom),
        SCMP_SYS(accept), SCMP_SYS(sendmsg), SCMP_SYS(recvmsg), SCMP_SYS(shutdown),
        SCMP_SYS(getpid), SCMP_SYS(gettid), SCMP_SYS(getuid), SCMP_SYS(geteuid),
        SCMP_SYS(getgid), SCMP_SYS(getegid), SCMP_SYS(arch_prctl), SCMP_SYS(exit_group)
    };
    for (int sys : whitelist) {
        if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, sys, 0) < 0) {
            perror("seccomp_rule_add");
            seccomp_release(ctx);
            _exit(1);
        }
    }

    if (seccomp_load(ctx) < 0) {
        perror("seccomp_load");
        seccomp_release(ctx);
        _exit(1);
    }
    seccomp_release(ctx);
}

// ---------------------------------------------------------------------
// Enforce a hard memory limit via cgroup v2 (optional, read from env)
// ---------------------------------------------------------------------
static void enforce_memory_limit(size_t limit_bytes) {
    const char *cgroup_root = "/sys/fs/cgroup";
    std::ofstream mem_max(std::string(cgroup_root) + "/memory.max");
    if (!mem_max) {
        std::cerr << "cgroup v2 not available – memory limit not enforced\n";
        return;
    }
    mem_max << limit_bytes << "\n";
}

// ---------------------------------------------------------------------
// Child‑process entry point – sets up the sandbox and returns to caller
// ---------------------------------------------------------------------
int sandbox_init(const Config& cfg) {
    // 1️⃣ Unshare namespaces (user, mount, pid, net)
    if (unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET) != 0) {
        perror("unshare");
        return 1;
    }

    // 2️⃣ Remount root as read‑only (best‑effort)
    if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
        perror("mount private");
    }
    if (mount(nullptr, "/", nullptr, MS_REMOUNT | MS_RDONLY, nullptr) != 0) {
        // not fatal – continue
    }

    // 3️⃣ Drop privileges
    drop_privileges();

    // 4️⃣ Install seccomp filter
    install_seccomp();

    // 5️⃣ Apply optional memory limit
    if (cfg.memory_limit_bytes) enforce_memory_limit(cfg.memory_limit_bytes);

    // 6️⃣ Success – continue execution in the same process
    return 0;
}
