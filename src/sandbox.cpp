#include "sandbox.hpp"

#include <stdexcept>

#ifdef LISA_ENABLE_SANDBOX

#include "utils.hpp"

#include <fcntl.h>
#include <linux/capability.h>
#include <linux/landlock.h>
#include <sched.h>
#include <seccomp.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <array>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define LISA_UNDER_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define LISA_UNDER_SANITIZER 1
#endif
#endif

namespace lisa {

namespace {

constexpr rlim_t kMaxOpenFiles = 64;
constexpr const char* kPidsMax = "1024";
constexpr const char* kHostnameText = "lisa-worker";
constexpr int kMaxCapabilityProbe = 63;

int g_landlock_abi = 0;

// Errors that mean "the host does not let unprivileged processes use namespaces".
const char* const kNamespaceHint =
    " (unprivileged user namespaces look blocked: on Ubuntu 23.10+ run "
    "'sysctl kernel.apparmor_restrict_unprivileged_userns=0' or add an AppArmor profile "
    "for lisa; in Docker run with '--security-opt seccomp=unconfined --security-opt "
    "apparmor=unconfined'; or set sandbox_enabled: false)";

[[noreturn]] void fail(const std::string& what, bool namespace_hint = false) {
    std::string message = "sandbox: " + what + ": " + std::strerror(errno);
    if (namespace_hint && (errno == EPERM || errno == EACCES || errno == ENOSPC)) {
        message += kNamespaceHint;
    }
    throw std::runtime_error(message);
}

void write_file(const std::string& path, const std::string& content, bool namespace_hint = false) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        fail("open " + path, namespace_hint);
    }
    const ssize_t n = ::write(fd, content.data(), content.size());
    const int saved = errno;
    ::close(fd);
    if (n != static_cast<ssize_t>(content.size())) {
        errno = saved;
        fail("write " + path, namespace_hint);
    }
}

bool is_directory(const char* path) {
    struct stat st {};
    return ::stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

// --- cgroup v2 --------------------------------------------------------------------------

void join_cgroup(const std::string& parent, std::uint64_t memory_limit) {
    if (::access((parent + "/cgroup.controllers").c_str(), F_OK) != 0) {
        throw std::runtime_error("sandbox: sandbox_cgroup '" + parent +
                                 "' is not a cgroup v2 directory");
    }
    const std::string dir = parent + "/lisa-worker-" + std::to_string(::getpid());
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        fail("create cgroup " + dir);
    }
    // Fail closed: a configured cgroup whose controllers are not delegated is an error.
    if (memory_limit > 0) {
        write_file(dir + "/memory.max", std::to_string(memory_limit));
        if (::access((dir + "/memory.swap.max").c_str(), F_OK) == 0) {
            write_file(dir + "/memory.swap.max", "0");
        }
    }
    write_file(dir + "/pids.max", kPidsMax);
    write_file(dir + "/cgroup.procs", std::to_string(::getpid()));
}

// --- mounts -----------------------------------------------------------------------------

// Mount flags a bind remount must keep: the kernel refuses to change locked flags of a
// mount inherited from a more privileged user namespace.
unsigned long inherited_mount_flags(int fd) {
    struct statvfs sv {};
    if (::fstatvfs(fd, &sv) != 0) {
        fail("fstatvfs model");
    }
    unsigned long flags = 0;
    if ((sv.f_flag & ST_NOATIME) != 0U) {
        flags |= MS_NOATIME;
    }
    if ((sv.f_flag & ST_NODIRATIME) != 0U) {
        flags |= MS_NODIRATIME;
    }
    if ((sv.f_flag & ST_RELATIME) != 0U) {
        flags |= MS_RELATIME;
    }
    return flags;
}

// The verified descriptor's mount belongs to the parent mount namespace, which the kernel
// will not bind from. Reopen its path here and require the very same inode.
int reopen_in_namespace(int model_fd) {
    std::array<char, PATH_MAX> link{};
    const std::string fd_link = "/proc/self/fd/" + std::to_string(model_fd);
    const ssize_t length = ::readlink(fd_link.c_str(), link.data(), link.size() - 1);
    if (length <= 0) {
        fail("resolve the model path");
    }
    const std::string path(link.data(), static_cast<std::size_t>(length));
    const int fd = ::open(path.c_str(), O_PATH | O_CLOEXEC);
    if (fd < 0) {
        fail("reopen the model file");
    }
    struct stat opened {};
    struct stat reopened {};
    if (::fstat(model_fd, &opened) != 0 || ::fstat(fd, &reopened) != 0 ||
        opened.st_dev != reopened.st_dev || opened.st_ino != reopened.st_ino) {
        ::close(fd);
        throw std::runtime_error("sandbox: the model file was replaced after it was opened");
    }
    return fd;
}

void build_root(int model_fd) {
    if (::mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
        fail("make mounts private", true);
    }
    const int model_in_ns = reopen_in_namespace(model_fd);

    const char* root = nullptr;
    for (const char* candidate : {"/tmp", "/mnt", "/run", "/var/tmp"}) {
        if (is_directory(candidate)) {
            root = candidate;
            break;
        }
    }
    if (root == nullptr) {
        throw std::runtime_error("sandbox: no directory to mount the worker root on");
    }
    if (::mount("lisa-root", root, "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC,
                "size=16k,nr_inodes=8,mode=0755") != 0) {
        fail(std::string("mount tmpfs on ") + root, true);
    }

    const std::string target = std::string(root) + std::string(kSandboxModelPath);
    const int placeholder = ::open(target.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0444);
    if (placeholder < 0) {
        fail("create " + target);
    }
    ::close(placeholder);

    // Bind through a descriptor opened in this namespace (resolved before the tmpfs
    // could hide the path); it was checked to be the verified file.
    const std::string source = "/proc/self/fd/" + std::to_string(model_in_ns);
    if (::mount(source.c_str(), target.c_str(), nullptr, MS_BIND, nullptr) != 0) {
        fail("bind the model file", true);
    }
    ::close(model_in_ns);
    const unsigned long ro_flags = MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV |
                                   MS_NOEXEC | inherited_mount_flags(model_fd);
    if (::mount(nullptr, target.c_str(), nullptr, ro_flags, nullptr) != 0) {
        fail("make the model mount read-only");
    }

    // Switch to the new root and detach the old one completely.
    if (::chdir(root) != 0) {
        fail(std::string("chdir ") + root);
    }
    if (::syscall(SYS_pivot_root, ".", ".") != 0) {
        fail("pivot_root");
    }
    if (::umount2(".", MNT_DETACH) != 0) {
        fail("detach the old root");
    }
    if (::chdir("/") != 0) {
        fail("chdir /");
    }
    if (::mount(nullptr, "/", nullptr, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC,
                nullptr) != 0) {
        fail("make the root read-only");
    }
}

// --- PID namespace reaper -----------------------------------------------------------------

// Seccomp for the reaper: it only waits for the worker and exits.
void install_reaper_seccomp() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (ctx == nullptr) {
        _exit(1);
    }
    for (const int nr :
         {SCMP_SYS(wait4), SCMP_SYS(waitid), SCMP_SYS(exit), SCMP_SYS(exit_group),
          SCMP_SYS(rt_sigreturn), SCMP_SYS(rt_sigprocmask), SCMP_SYS(restart_syscall)}) {
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, nr, 0);
    }
    if (seccomp_load(ctx) != 0) {
        _exit(1);
    }
    seccomp_release(ctx);
}

[[noreturn]] void run_reaper(pid_t worker) {
    // Hold nothing: the broker must see EOF on the channel as soon as the worker dies.
    ::close_range(0, ~0U, 0);
    install_reaper_seccomp();
    int status = 0;
    while (::waitpid(worker, &status, 0) < 0) {
        if (errno != EINTR) {
            _exit(1);
        }
    }
    if (WIFSIGNALED(status)) {
        _exit(128 + WTERMSIG(status));
    }
    _exit(WIFEXITED(status) ? WEXITSTATUS(status) : 1);
}

// --- privileges -------------------------------------------------------------------------

void set_limit(int resource, rlim_t value) {
    const rlimit limit{value, value};
    if (::setrlimit(resource, &limit) != 0) {
        fail("setrlimit");
    }
}

void drop_capabilities() {
    for (int cap = 0; cap <= kMaxCapabilityProbe; ++cap) {
        if (::prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0 && errno != EINVAL) {
            fail("drop bounding capabilities");
        }
    }
    if (::prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0) {
        fail("clear ambient capabilities");
    }
    __user_cap_header_struct header{_LINUX_CAPABILITY_VERSION_3, 0};
    std::array<__user_cap_data_struct, 2> data{};
    if (::syscall(SYS_capset, &header, data.data()) != 0) {
        fail("capset");
    }
}

// --- Landlock ---------------------------------------------------------------------------

// Kernel ABI layout of landlock_ruleset_attr (newer than some system headers).
struct LandlockRulesetAttr {
    std::uint64_t handled_access_fs;
    std::uint64_t handled_access_net;
    std::uint64_t scoped;
};

constexpr std::uint64_t kLandlockFsV1 = (1ULL << 13U) - 1;                // EXECUTE .. MAKE_SYM
constexpr std::uint64_t kLandlockFsRefer = 1ULL << 13U;                   // ABI 2
constexpr std::uint64_t kLandlockFsTruncate = 1ULL << 14U;                // ABI 3
constexpr std::uint64_t kLandlockFsIoctlDev = 1ULL << 15U;                // ABI 5
constexpr std::uint64_t kLandlockNetAll = (1ULL << 0U) | (1ULL << 1U);    // ABI 4: bind+connect
constexpr std::uint64_t kLandlockScopeAll = (1ULL << 0U) | (1ULL << 1U);  // ABI 6: unix+signal
constexpr std::uint64_t kLandlockReadFile = 1ULL << 2U;

// Denies every filesystem access except reading `read_only_file`, every TCP bind/connect,
// abstract Unix sockets and signals to processes outside the sandbox (as far as the
// running kernel's Landlock ABI allows). Returns the ABI used, 0 when unavailable.
int apply_landlock(const char* read_only_file) {
    const long abi =
        ::syscall(SYS_landlock_create_ruleset, nullptr, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi <= 0) {
        return 0;
    }
    LandlockRulesetAttr attr{};
    attr.handled_access_fs = kLandlockFsV1;
    std::size_t attr_size = offsetof(LandlockRulesetAttr, handled_access_net);
    if (abi >= 2) {
        attr.handled_access_fs |= kLandlockFsRefer;
    }
    if (abi >= 3) {
        attr.handled_access_fs |= kLandlockFsTruncate;
    }
    if (abi >= 4) {
        attr.handled_access_net = kLandlockNetAll;
        attr_size = offsetof(LandlockRulesetAttr, scoped);
    }
    if (abi >= 5) {
        attr.handled_access_fs |= kLandlockFsIoctlDev;
    }
    if (abi >= 6) {
        attr.scoped = kLandlockScopeAll;
        attr_size = sizeof(LandlockRulesetAttr);
    }

    const int ruleset =
        static_cast<int>(::syscall(SYS_landlock_create_ruleset, &attr, attr_size, 0));
    if (ruleset < 0) {
        fail("landlock_create_ruleset");
    }
    const int file = ::open(read_only_file, O_PATH | O_CLOEXEC);
    if (file < 0) {
        ::close(ruleset);
        fail(std::string("open ") + read_only_file);
    }
    landlock_path_beneath_attr rule{};
    rule.allowed_access = kLandlockReadFile;
    rule.parent_fd = file;
    const long added =
        ::syscall(SYS_landlock_add_rule, ruleset, LANDLOCK_RULE_PATH_BENEATH, &rule, 0);
    ::close(file);
    if (added != 0 || ::syscall(SYS_landlock_restrict_self, ruleset, 0) != 0) {
        const int saved = errno;
        ::close(ruleset);
        errno = saved;
        fail("landlock");
    }
    ::close(ruleset);
    return static_cast<int>(abi);
}

// --- seccomp ----------------------------------------------------------------------------

struct FilterDeleter {
    void operator()(void* ctx) const noexcept { seccomp_release(ctx); }
};

void allow(scmp_filter_ctx ctx, int nr) {
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, nr, 0) != 0) {
        throw std::runtime_error("sandbox: seccomp_rule_add failed");
    }
}

void deny_with(scmp_filter_ctx ctx, int nr, int error) {
    if (seccomp_rule_add(ctx, SCMP_ACT_ERRNO(static_cast<std::uint32_t>(error)), nr, 0) != 0) {
        throw std::runtime_error("sandbox: seccomp_rule_add failed");
    }
}

}  // namespace

bool sandbox_available() noexcept {
    return true;
}

int landlock_abi_in_effect() noexcept {
    return g_landlock_abi;
}

void enter_worker_sandbox(const SandboxPolicy& policy, int model_fd) {
    // Die with the broker.
    if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
        fail("PR_SET_PDEATHSIG");
    }
    if (!policy.cgroup_parent.empty()) {
        join_cgroup(policy.cgroup_parent, policy.memory_limit_bytes);
    }

    const uid_t uid = ::getuid();
    const gid_t gid = ::getgid();
    const int namespaces = CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET | CLONE_NEWIPC |
                           CLONE_NEWUTS | CLONE_NEWCGROUP | CLONE_NEWPID;
    if (::unshare(namespaces) != 0) {
        fail("unshare namespaces", true);
    }
    write_file("/proc/self/setgroups", "deny", true);
    write_file("/proc/self/uid_map", "0 " + std::to_string(uid) + " 1", true);
    write_file("/proc/self/gid_map", "0 " + std::to_string(gid) + " 1", true);

    build_root(model_fd);
    if (::sethostname(kHostnameText, std::strlen(kHostnameText)) != 0) {
        fail("sethostname");
    }

    // unshare(CLONE_NEWPID) only applies to children: the worker continues as PID 1 of
    // the new namespace, this process stays outside and reaps it.
    const pid_t worker = ::fork();
    if (worker < 0) {
        fail("fork into the PID namespace");
    }
    if (worker > 0) {
        run_reaper(worker);
    }
    if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
        fail("PR_SET_PDEATHSIG");
    }

    set_limit(RLIMIT_CORE, 0);
    set_limit(RLIMIT_NOFILE, kMaxOpenFiles);
#ifndef LISA_UNDER_SANITIZER  // sanitizer shadow memory counts against RLIMIT_DATA
    if (policy.memory_limit_bytes > 0) {
        set_limit(RLIMIT_DATA, static_cast<rlim_t>(policy.memory_limit_bytes));
    }
#endif
    if (::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 ||
        ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        fail("prctl");
    }
    g_landlock_abi = apply_landlock(std::string(kSandboxModelPath).c_str());
    if (g_landlock_abi == 0) {
        log_warn(
            "Landlock is not available on this kernel; the sandbox relies on namespaces "
            "and seccomp only");
    }
    drop_capabilities();
}

void install_seccomp(SeccompStage stage, bool log_only) {
    const std::unique_ptr<void, FilterDeleter> filter(
        seccomp_init(log_only ? SCMP_ACT_LOG : SCMP_ACT_KILL_PROCESS));
    scmp_filter_ctx ctx = filter.get();
    // no_new_privs is already set by enter_worker_sandbox, so libseccomp need not (and
    // must not, under the load filter) call prctl itself.
    if (ctx == nullptr || seccomp_attr_set(ctx, SCMP_FLTATR_CTL_TSYNC, 1) != 0 ||
        seccomp_attr_set(ctx, SCMP_FLTATR_CTL_NNP, 0) != 0) {
        throw std::runtime_error("sandbox: cannot initialise seccomp");
    }

    // Needed for computing, threading, memory management and talking to the broker.
    for (const int nr : {
             SCMP_SYS(read),
             SCMP_SYS(write),
             SCMP_SYS(readv),
             SCMP_SYS(writev),
             SCMP_SYS(close),
             SCMP_SYS(recvfrom),
             SCMP_SYS(sendto),
             SCMP_SYS(poll),
             SCMP_SYS(ppoll),
             SCMP_SYS(mmap),
             SCMP_SYS(munmap),
             SCMP_SYS(mprotect),
             SCMP_SYS(mremap),
             SCMP_SYS(madvise),
             SCMP_SYS(brk),
             SCMP_SYS(futex),
             SCMP_SYS(set_robust_list),
             SCMP_SYS(rseq),
             SCMP_SYS(rt_sigaction),
             SCMP_SYS(rt_sigprocmask),
             SCMP_SYS(rt_sigreturn),
             SCMP_SYS(sigaltstack),
             SCMP_SYS(exit),
             SCMP_SYS(exit_group),
             SCMP_SYS(gettid),
             SCMP_SYS(getpid),
             SCMP_SYS(sched_yield),
             SCMP_SYS(sched_getaffinity),
             SCMP_SYS(nanosleep),
             SCMP_SYS(clock_nanosleep),
             SCMP_SYS(clock_gettime),
             SCMP_SYS(gettimeofday),
             SCMP_SYS(getrandom),
             SCMP_SYS(restart_syscall),
             SCMP_SYS(fstat),
             SCMP_SYS(newfstatat),
             SCMP_SYS(lseek),
             SCMP_SYS(getrusage),
             SCMP_SYS(prlimit64),
         }) {
        allow(ctx, nr);
    }

    // Threads only: clone3's flags cannot be inspected, so make glibc fall back to clone
    // and allow clone only with CLONE_THREAD (no new processes or namespaces).
    deny_with(ctx, SCMP_SYS(clone3), ENOSYS);
    // Explicit structs: the SCMP_A0() macros are C compound literals.
    const scmp_arg_cmp thread_flag{0, SCMP_CMP_MASKED_EQ, CLONE_THREAD, CLONE_THREAD};
    if (seccomp_rule_add_array(ctx, SCMP_ACT_ALLOW, SCMP_SYS(clone), 1, &thread_flag) != 0) {
        throw std::runtime_error("sandbox: seccomp_rule_add(clone) failed");
    }
    // Self-signals only (abort(), thread cancellation).
    const scmp_arg_cmp own_pid{0, SCMP_CMP_EQ, static_cast<scmp_datum_t>(::getpid()), 0};
    if (seccomp_rule_add_array(ctx, SCMP_ACT_ALLOW, SCMP_SYS(tgkill), 1, &own_pid) != 0) {
        throw std::runtime_error("sandbox: seccomp_rule_add(tgkill) failed");
    }
    deny_with(ctx, SCMP_SYS(ioctl), ENOTTY);  // e.g. isatty() probes
#ifdef LISA_UNDER_SANITIZER
    allow(ctx, SCMP_SYS(pipe2));  // AddressSanitizer probes memory through a pipe
#endif

    if (stage == SeccompStage::load) {
        // Opening and mapping the model, and installing the serve-stage filter.
        for (const int nr :
             {SCMP_SYS(openat), SCMP_SYS(statx), SCMP_SYS(pread64), SCMP_SYS(fcntl),
              SCMP_SYS(readlinkat), SCMP_SYS(getdents64), SCMP_SYS(faccessat2), SCMP_SYS(fstatfs),
              SCMP_SYS(sysinfo), SCMP_SYS(fadvise64), SCMP_SYS(seccomp)}) {
            allow(ctx, nr);
        }
    } else {
        // Library probes of /proc or /sys degrade gracefully instead of killing the worker.
        deny_with(ctx, SCMP_SYS(openat), EACCES);
    }

    if (seccomp_load(ctx) != 0) {
        throw std::runtime_error("sandbox: seccomp_load failed");
    }
}

}  // namespace lisa

#else  // !LISA_ENABLE_SANDBOX

namespace lisa {

bool sandbox_available() noexcept {
    return false;
}

int landlock_abi_in_effect() noexcept {
    return 0;
}

void enter_worker_sandbox(const SandboxPolicy& /*policy*/, int /*model_fd*/) {
    throw std::runtime_error("this binary was built with LISA_SANDBOX=OFF");
}

void install_seccomp(SeccompStage /*stage*/, bool /*log_only*/) {
    throw std::runtime_error("this binary was built with LISA_SANDBOX=OFF");
}

}  // namespace lisa

#endif
