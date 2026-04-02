#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class BPETokenizer {
public:
    BPETokenizer();
    void load(const std::string& vocab_path, const std::string& merges_path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    size_t vocab_size() const { return token_to_id_.size(); }
    
private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int> bpe_ranks_;
    
    std::string bpe(const std::string& token) const;
    std::vector<std::string> byte_encode(const std::string& text) const;
    std::string byte_decode(const std::vector<std::string>& tokens) const;
    static std::unordered_map<char, std::string> build_byte_encoder();
    static std::unordered_map<std::string, char> build_byte_decoder();
};
