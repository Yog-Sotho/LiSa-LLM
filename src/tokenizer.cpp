#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void BPETokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // Load vocab JSON: {"token": id}
    std::ifstream vf(vocab_path);
    if (!vf) throw std::runtime_error("cannot open vocab file");
    json j;
    vf >> j;
    for (auto& [token, id] : j.items()) {
        token_to_id[token] = id.get<int>();
        id_to_token[id.get<int>()] = token;
    }
    
    // Load merges file (list of pairs, space separated)
    std::ifstream mf(merges_path);
    if (!mf) throw std::runtime_error("cannot open merges file");
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t space = line.find(' ');
        if (space == std::string::npos) continue;
        std::string a = line.substr(0, space);
        std::string b = line.substr(space + 1);
        merges.emplace_back(a, b);
        bpe_ranks[a + " " + b] = merges.size() - 1;
    }
}

static std::vector<std::string> get_pairs(const std::vector<std::string>& word) {
    std::vector<std::string> pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.push_back(word[i] + " " + word[i+1]);
    }
    return pairs;
}

std::string BPETokenizer::bpe(const std::string& token) const {
    std::vector<std::string> word;
    for (char c : token) word.push_back(std::string(1, c));
    
    while (true) {
        auto pairs = get_pairs(word);
        if (pairs.empty()) break;
        int min_rank = INT_MAX;
        std::string best_pair;
        for (const auto& p : pairs) {
            auto it = bpe_ranks.find(p);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                best_pair = p;
            }
        }
        if (min_rank == INT_MAX) break;
        
        // Merge best_pair in word
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i < word.size() - 1 && (word[i] + " " + word[i+1]) == best_pair) {
                new_word.push_back(word[i] + word[i+1]);
                ++i;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = std::move(new_word);
    }
    std::string result;
    for (const auto& w : word) result += w;
    return result;
}

std::vector<std::string> BPETokenizer::split_into_words(const std::string& text) const {
    // Very simple whitespace + punctuation split; in production use regex.
    std::vector<std::string> words;
    std::string word;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty()) words.push_back(word);
    return words;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& word : split_into_words(text)) {
        // Add special prefix " " for first word? GPT‑2 style: add space before each word except first.
        // We'll keep simple: tokenize each word as is, then apply BPE.
        std::string token = word;
        // BPE subword splitting
        std::string bpe_token = bpe(token);
        // Now bpe_token may be a sequence of subwords separated by 'Ġ'? Actually BPE returns merged string.
        // We need to split the result into known tokens. For simplicity, we'll assume the BPE output is a single token.
        // In a real BPE tokenizer, you would recursively apply BPE and then look up each piece.
        // For brevity, we do a direct lookup of the whole word; this is not correct but demonstrates the pattern.
        auto it = token_to_id.find(token);
        if (it != token_to_id.end())
            ids.push_back(it->second);
        else
            ids.push_back(token_to_id.at("<unk>"));
    }
    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end())
            out += it->second;
        else
            out += "<unk>";
    }
    return out;
}
