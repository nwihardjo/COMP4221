//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_SHARED_UTILS_HPP
#define COMP4221_2019Q1_A3_SHARED_UTILS_HPP
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class frequent_token_collector {
public:

  /**
   * +1 to a token's occurence
   * \param token
   */
  void add_occurence(const std::string &token) {
    if (counts.count(token) > 0) {
      counts[token]++;
    } else {
      counts[token] = 1;
    }
  }

  /**
   * get the X most frequent tokens.
   * if the tokens seens are less than X, all tokens will be returned.
   * \param size top X
   * \return the list of tokens, ordered from most frequent to most infrequent.
   */
  std::vector<std::string> list_frequent_tokens(unsigned size) const {
    std::vector<std::pair<std::string, unsigned>> pairs;
    for (const auto &p:counts) {
      pairs.emplace_back(p);
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto &x, const auto &y) { return x.second > y.second; });
    if (pairs.size() > size) pairs.resize(size);
    std::vector<std::string> ret;
    ret.reserve(pairs.size());
    for (const auto &p:pairs) {
      ret.push_back(p.first);
    }
    return ret;
  }

  void print_summary() const {
    std::vector<std::pair<std::string, unsigned>> pairs;
    for (const auto &p:counts) {
      pairs.emplace_back(p);
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto &x, const auto &y) { return x.second > y.second; });
    unsigned i=0;
    for(const auto &[token, count]:pairs) {
      cout << i << ". " << token << " "<< count <<endl;
      i++;
    }
  }

private:
  std::unordered_map<std::string, unsigned> counts;
};

constexpr unsigned MAX_VOCAB_SIZE = 1000;

/**
 * a helper function to collect vocabulary from training data
 * collect distinct symbols from a list of symbols that may contain duplicates
 * \param symbols a list of symbols
 * \return the distinct list
 */
inline vector<string> collect_vocab(const vector<string> &symbols) {
  frequent_token_collector collector;
  for(const auto &symbol:symbols) {
    collector.add_occurence(symbol);
  }
  return collector.list_frequent_tokens(MAX_VOCAB_SIZE);
}

/**
 * a helper function to collect vocabulary from training data
 * collect distinct symbols from a matrix of symbols that may contain duplicates
 * \param symbol_matrix a matrix of symbols
 * \return the distince list
 */
inline vector<string> collect_vocab_from_symbol_matrix(const vector<vector<string>> &symbol_matrix) {
  vector<string> symbols;
  for (const auto &row:symbol_matrix) {
    copy(row.begin(), row.end(), back_inserter(symbols));
  }
  return collect_vocab(symbols);
}

#endif //COMP4221_2019Q1_A3_SHARED_UTILS_HPP
