#include "sandbox.hpp"
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <seccomp.h>
#include <sched.h>
#include <cstring>
#include <fstream>
#include <cerrno>
#include <cstdlib>

#ifdef LISA_ENABLE_SANDBOX

static void drop_privileges() {
    if (setgid(65534) != 0 || setuid(65534) != 0) {
        perror("drop_privileges");
        _exit(1);
    }
}

static void install_seccomp() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL);
    if (!ctx) { perror("seccomp_init"); _exit(1); }
    const int allowed[] = {
        SCMP_SYS(read), SCMP_SYS(write), SCMP_SYS(close), SCMP_SYS(fstat),
        SCMP_SYS(lseek), SCMP_SYS(mmap), SCMP_SYS(munmap), SCMP_SYS(brk),
        SCMP_SYS(rt_sigaction), SCMP_SYS(rt_sigprocmask), SCMP_SYS(exit_group),
        SCMP_SYS(getpid), SCMP_SYS(gettid), SCMP_SYS(getuid), SCMP_SYS(geteuid),
        SCMP_SYS(getgid), SCMP_SYS(getegid), SCMP_SYS(arch_prctl),
        SCMP_SYS(socket), SCMP_SYS(bind), SCMP_SYS(listen), SCMP_SYS(accept4),
        SCMP_SYS(recvfrom), SCMP_SYS(sendto), SCMP_SYS(epoll_create1),
        SCMP_SYS(epoll_ctl), SCMP_SYS(epoll_wait), SCMP_SYS(getsockname),
        SCMP_SYS(setsockopt), SCMP_SYS(getsockopt), SCMP_SYS(connect)
    };
    for (int sys : allowed) {
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

static void setup_memory_limit(size_t limit_bytes) {
    pid_t pid = getpid();
    std::string cgroup_path = "/sys/fs/cgroup/lisa_" + std::to_string(pid);
    if (mkdir(cgroup_path.c_str(), 0755) != 0 && errno != EEXIST) {
        perror("mkdir cgroup");
        return;
    }
    std::ofstream procs(cgroup_path + "/cgroup.procs");
    if (procs) {
        procs << pid << std::endl;
        procs.close();
        std::ofstream max(cgroup_path + "/memory.max");
        if (max) max << limit_bytes << std::endl;
    }
}

static void pivot_root_sandbox() {
    char tmpdir[] = "/tmp/lisa_sandbox_XXXXXX";
    if (!mkdtemp(tmpdir)) { perror("mkdtemp"); _exit(1); }
    if (mount(tmpdir, tmpdir, "tmpfs", MS_NOSUID | MS_NODEV, nullptr) != 0) {
        perror("mount tmpfs"); _exit(1);
    }
    if (chdir(tmpdir) != 0) { perror("chdir"); _exit(1); }
    if (pivot_root(".", ".") != 0) { perror("pivot_root"); _exit(1); }
    if (chroot(".") != 0) { perror("chroot"); _exit(1); }
    if (umount2("/", MNT_DETACH) != 0) { /* ignore */ }
}

int sandbox_init(const Config& cfg) {
    if (unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET) != 0) {
        perror("unshare"); return 1;
    }
    if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
        perror("mount private");
    }
    if (cfg.memory_limit_bytes > 0) setup_memory_limit(cfg.memory_limit_bytes);
    pivot_root_sandbox();
    drop_privileges();
    install_seccomp();
    return 0;
}

#else
int sandbox_init(const Config& cfg) { return 0; }
#endif
