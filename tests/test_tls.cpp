// TLS hardening and mutual TLS, with certificates generated at run time.

#include "fake_engine.hpp"
#include "server.hpp"

#include <httplib.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

namespace fs = std::filesystem;

struct PkeyDeleter {
    void operator()(EVP_PKEY* key) const { EVP_PKEY_free(key); }
};
struct X509Deleter {
    void operator()(X509* cert) const { X509_free(cert); }
};
using Pkey = std::unique_ptr<EVP_PKEY, PkeyDeleter>;
using Cert = std::unique_ptr<X509, X509Deleter>;

Pkey make_key() {
    Pkey key(EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256"));
    if (!key) {
        throw std::runtime_error("key generation failed");
    }
    return key;
}

void add_extension(X509* cert, X509* issuer, int nid, const char* value) {
    X509V3_CTX ctx;
    X509V3_set_ctx_nodb(&ctx);
    X509V3_set_ctx(&ctx, issuer, cert, nullptr, nullptr, 0);
    X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
    if (ext == nullptr || X509_add_ext(cert, ext, -1) != 1) {
        throw std::runtime_error("cannot add certificate extension");
    }
    X509_EXTENSION_free(ext);
}

// A certificate for `key`, signed by `issuer` (self-signed when null).
Cert make_cert(EVP_PKEY* key, const char* common_name, bool ca, X509* issuer = nullptr,
               EVP_PKEY* issuer_key = nullptr) {
    static long serial = 1;
    Cert cert(X509_new());
    X509_set_version(cert.get(), 2);
    ASN1_INTEGER_set(X509_get_serialNumber(cert.get()), serial++);
    X509_gmtime_adj(X509_getm_notBefore(cert.get()), -60);
    X509_gmtime_adj(X509_getm_notAfter(cert.get()), 3600);
    X509_set_pubkey(cert.get(), key);
    X509_NAME* name = X509_get_subject_name(cert.get());
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                               reinterpret_cast<const unsigned char*>(common_name), -1, -1, 0);
    X509* signer = issuer != nullptr ? issuer : cert.get();
    X509_set_issuer_name(cert.get(), X509_get_subject_name(signer));
    if (ca) {
        add_extension(cert.get(), signer, NID_basic_constraints, "critical,CA:TRUE");
        add_extension(cert.get(), signer, NID_key_usage, "critical,keyCertSign,cRLSign");
    } else {
        add_extension(cert.get(), signer, NID_basic_constraints, "critical,CA:FALSE");
        add_extension(cert.get(), signer, NID_subject_alt_name, "IP:127.0.0.1");
    }
    if (X509_sign(cert.get(), issuer_key != nullptr ? issuer_key : key, EVP_sha256()) == 0) {
        throw std::runtime_error("signing failed");
    }
    return cert;
}

void write_pem(const fs::path& path, X509* cert) {
    FILE* file = std::fopen(path.c_str(), "w");
    REQUIRE(file != nullptr);
    PEM_write_X509(file, cert);
    (void)std::fclose(file);
}

void write_pem(const fs::path& path, EVP_PKEY* key) {
    FILE* file = std::fopen(path.c_str(), "w");
    REQUIRE(file != nullptr);
    PEM_write_PrivateKey(file, key, nullptr, nullptr, 0, nullptr, nullptr);
    (void)std::fclose(file);
}

// A CA, a server certificate and two client certificates (one CA-signed, one not).
class Pki {
public:
    Pki() : dir_(fs::temp_directory_path() / ("lisa_tls_" + std::to_string(::getpid()))) {
        fs::create_directories(dir_);
        const Pkey ca_key = make_key();
        const Cert ca = make_cert(ca_key.get(), "LiSa test CA", true);
        write_pem(path("ca.pem"), ca.get());

        const Pkey server_key = make_key();
        write_pem(path("server.pem"),
                  make_cert(server_key.get(), "127.0.0.1", false, ca.get(), ca_key.get()).get());
        write_pem(path("server.key"), server_key.get());

        const Pkey client_key = make_key();
        write_pem(path("client.pem"),
                  make_cert(client_key.get(), "client", false, ca.get(), ca_key.get()).get());
        write_pem(path("client.key"), client_key.get());

        const Pkey rogue_key = make_key();
        write_pem(path("rogue.pem"), make_cert(rogue_key.get(), "rogue", false).get());
        write_pem(path("rogue.key"), rogue_key.get());
    }
    ~Pki() { fs::remove_all(dir_); }
    Pki(const Pki&) = delete;
    Pki& operator=(const Pki&) = delete;
    Pki(Pki&&) = delete;
    Pki& operator=(Pki&&) = delete;

    [[nodiscard]] std::string path(const char* name) const { return (dir_ / name).string(); }

private:
    fs::path dir_;
};

const Pki& pki() {
    static const Pki instance;
    return instance;
}

class TlsServer {
public:
    explicit TlsServer(lisa::Config cfg) : server_(engine_, std::move(cfg)) {
        server_.bind();
        thread_ = std::thread([this] { server_.run(); });
    }
    ~TlsServer() {
        server_.stop();
        thread_.join();
    }
    TlsServer(const TlsServer&) = delete;
    TlsServer& operator=(const TlsServer&) = delete;
    TlsServer(TlsServer&&) = delete;
    TlsServer& operator=(TlsServer&&) = delete;

    [[nodiscard]] int port() const { return server_.port(); }

private:
    lisa::test::FakeEngine engine_;
    lisa::HttpServer server_;
    std::thread thread_;
};

lisa::Config tls_config() {
    lisa::Config cfg;
    cfg.model_path = "fake.gguf";
    cfg.listen_addr = "127.0.0.1";
    cfg.listen_port = 0;
    cfg.enable_tls = true;
    cfg.cert_file = pki().path("server.pem");
    cfg.key_file = pki().path("server.key");
    return cfg;
}

std::unique_ptr<httplib::SSLClient> client(int port, const char* cert = nullptr,
                                           const char* key = nullptr) {
    auto c = cert != nullptr ? std::make_unique<httplib::SSLClient>(
                                   "127.0.0.1", port, pki().path(cert), pki().path(key))
                             : std::make_unique<httplib::SSLClient>("127.0.0.1", port);
    c->set_ca_cert_path(pki().path("ca.pem"));
    c->enable_server_certificate_verification(true);
    c->set_connection_timeout(5, 0);
    return c;
}

SSL_CTX* context_of(httplib::SSLClient& c) {
    return static_cast<SSL_CTX*>(c.tls_context());
}

}  // namespace

TEST_CASE("TLS: verified HTTPS with HSTS", "[tls]") {
    const TlsServer server(tls_config());
    auto c = client(server.port());
    const auto res = c->Get("/healthz");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->get_header_value("Strict-Transport-Security") == "max-age=31536000");
}

TEST_CASE("TLS: only forward-secret AEAD suites on TLS 1.2", "[tls]") {
    const TlsServer server(tls_config());

    auto aead = client(server.port());
    REQUIRE(SSL_CTX_set_max_proto_version(context_of(*aead), TLS1_2_VERSION) == 1);
    REQUIRE(SSL_CTX_set_cipher_list(context_of(*aead), "ECDHE-ECDSA-AES128-GCM-SHA256") == 1);
    const auto ok = aead->Get("/healthz");
    REQUIRE(ok);
    CHECK(ok->status == 200);

    auto cbc = client(server.port());
    REQUIRE(SSL_CTX_set_max_proto_version(context_of(*cbc), TLS1_2_VERSION) == 1);
    REQUIRE(SSL_CTX_set_cipher_list(context_of(*cbc), "ECDHE-ECDSA-AES128-SHA256") == 1);
    CHECK_FALSE(cbc->Get("/healthz"));
}

TEST_CASE("TLS: tls_min_version 1.3 refuses TLS 1.2", "[tls]") {
    lisa::Config cfg = tls_config();
    cfg.tls_min_version = "1.3";
    const TlsServer server(cfg);

    auto old = client(server.port());
    REQUIRE(SSL_CTX_set_max_proto_version(context_of(*old), TLS1_2_VERSION) == 1);
    CHECK_FALSE(old->Get("/healthz"));

    auto modern = client(server.port());
    const auto res = modern->Get("/healthz");
    REQUIRE(res);
    CHECK(res->status == 200);
}

TEST_CASE("TLS: client_ca_file requires a client certificate from that CA", "[tls]") {
    lisa::Config cfg = tls_config();
    cfg.client_ca_file = pki().path("ca.pem");
    const TlsServer server(cfg);

    CHECK_FALSE(client(server.port())->Get("/healthz"));
    CHECK_FALSE(client(server.port(), "rogue.pem", "rogue.key")->Get("/healthz"));

    const auto res = client(server.port(), "client.pem", "client.key")->Get("/healthz");
    REQUIRE(res);
    CHECK(res->status == 200);
}

TEST_CASE("TLS: bad certificate files fail at startup", "[tls]") {
    lisa::test::FakeEngine engine;
    lisa::Config cfg = tls_config();
    cfg.cert_file = pki().path("missing.pem");
    CHECK_THROWS_AS(lisa::HttpServer(engine, cfg), std::runtime_error);
}
