#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class BPETokenizer {
public:
    BPETokenizer() = default;
    void load(const std::string& vocab_path, const std::string& merges_path);
    
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    
    size_t vocab_size() const { return token_to_id.size(); }
    
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges; // ordered pairs
    std::unordered_map<std::string, int> bpe_ranks;
    
    std::string bpe(const std::string& token) const;
    std::vector<std::string> split_into_words(const std::string& text) const;
};
